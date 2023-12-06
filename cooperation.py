# 本文件中所有的位置信息均使用grid编号，Lon及Lat仅表示X Y
# 本文件中所有的距离都采用km， 时间都采用min
# spatial division 为 aBeacon的方式:
# (t1.longitude * 100) - 12110))  (t1.latitude * 100) - 3070)
# slot = 1 min
from sklearn.linear_model import LinearRegression
import sys
import os
import numpy as np
import pyodbc
import math
import datetime
import random

np.seterr(divide='ignore', invalid='ignore')        # ignore np division and invalid
# 所有欧拉距离都×1.5，当作曼哈顿距离
distance_coefficient = 1.5

# variable parameters
parameterChanged = str(sys.argv[1])     # sys.argv[0]是被调用的python文件的名字
allowFreeDelivery = int(sys.argv[2])
deliveryProportion = float(sys.argv[3])
taxiProportion = float(sys.argv[4])
UAVPerBase = int(sys.argv[5])
baseNum = int(sys.argv[6])
taxiLeisureProbabilityMin = float(sys.argv[7])      # 在uav repositioning时设定最小的空车概率为0.7，若空车概率小于0.7则运力缺口+1
UAVEndurance = int(sys.argv[8])                 # UAV flight time
SecSpeed = int(sys.argv[9])
detourTimeLimit = float(sys.argv[10])
PTL = int(sys.argv[11])
day = int(sys.argv[12])
###

# time constraints
originTime_in = detourTimeLimit / 3         # mode c 的起点绕路时限    3   2
destTime_in = detourTimeLimit / 6           # mode c 的送达绕路时限    1   1
originTime_OD = detourTimeLimit / 3         # mode b 的起点绕路时限    3   2
destTime_OD = detourTimeLimit * 2 / 3       # mode b 的送达绕路时限    5   3
detourTimeLimitSimplified = 10       # 在uav repositioning时使用     10  5
####

# const parameters
slotLength = 1              # 每个slot一分钟
slotNum = 12 * 60 // slotLength
####

# dependent variables
taxiSpeedDefault = 0.5       # 单位为km/min， 0.5km/min = 30km/h
UAVSpeed = SecSpeed * 3.6 / 60      # 单位为 km/min
deliveryNum = -1            # 在 readDelivery 函数中修改
###

# impact of the parcel weight on the uav endurance
k = 0
b = 0
####

# results
allRepositionedUAVNum = 0
freeTaxiDeliveredNum = 0
detourTimeOriginList = []           # 仅记录每次taxi派送的起点绕路时长
###

taxiList = []
baseList = []                   # [x, y, uavNumber]
UAVList = []                    # UAVNo, baseNo, x, y, remaining energy, status (busy:0 / free:1),
                                # 接上 {freeSlot, backTobaseSlot, deliEndLon, deliEndLat} 花括号里的为deliveryUAV专属
deliveredPackageList = []       # 存放被成功配送的包裹列表， [delivery time, by who(0:taxi, 1:UAV), former delivery time]
undeliveredPackageList = []     # 存放没有被配送的包裹列表
taxiDetourList = []             # 记录配送该包裹出租车的总绕路时长

tripList = [[] for i in range(slotNum)]           # 分slot记录trip
estimated_tripList = [[] for i in range(slotNum)]           # 分slot记录trip
predictedDeliveryList = [[] for i in range(slotNum)]       # 分slot记录delivery
actualDeliveryList = [[] for i in range(slotNum)]       # 分slot记录delivery

gridBelongingMap = np.zeros((80,80),dtype=int)          # 记录每个grid归属的基站，80*80为地图
tripDepartureMap = np.zeros((slotNum, 80, 80), dtype=int)       # 记录每个时刻每个grid出发的taxi数
tripArrivalMap = np.zeros((slotNum, 80, 80), dtype=int)         # 记录每个时刻每个grid到达的taxi数


# tool func
def cal_distance(x1, y1, x2, y2):
    distance = math.sqrt(pow(x1-x2, 2) + pow(y1-y2, 2))
    return distance + random.uniform(0, 1.414)       # 距离计算完后再加上一个0,1.414的小数，作为对grid内距离的估计


# 本函数用于计算点到线段距离，p,a,b 依次为点及线段的两个顶点
# 且以上三点均为[0,0],[1,1]形式
def dis_point_to_seg_line(p, a, b):
    a = a[:2]           # 防止传入路径点有额外的feature，这里只取前两位
    b = b[:2]
    a, b, p = np.array(a), np.array(b), np.array(p)	 # trans to np.array
    d = np.divide(b - a, np.linalg.norm(b - a))     # normalized tangent vector
    s = np.dot(a - p, d)        # signed parallel distance components
    t = np.dot(p - b, d)
    h = np.maximum.reduce([s, t, 0])        # clamped parallel distance
    c = np.cross(p - a, d)      # perpendicular distance component
    return np.hypot(h, np.linalg.norm(c))


# 时间相减函数， 输入：两个字符串时间戳，如"12:12:12"，输出 前者 - 后者 的秒数
def time_sub(str1, str2):
    array1 = str1.split(":", 2)
    hour1 = array1[0]
    minute1 = array1[1]
    second1 = array1[2]
    time1 = int(hour1) * 3600 + int(minute1) * 60 + int(second1)
    array2 = str2.split(":", 2)
    hour2 = array2[0]
    minute2 = array2[1]
    second2 = array2[2]
    time2 = int(hour2) * 3600 + int(minute2) * 60 + int(second2)
    return time1-time2


def weightLinearFit():
    global k,b
    weightList = [0, 169.5, 233.9, 415.3, 584.7, 639, 1015.3]
    weightList = [weightList[i] / 1000 for i in range(len(weightList))]           # 单位化为kg
    enduranceList = [34, 31, 30, 29, 28, 27, 26]

    weight_train = np.array(weightList).reshape((len(weightList), 1))
    endurance_train = np.array(enduranceList).reshape((len(weightList), 1))

    linearModel = LinearRegression()
    linearModel.fit(weight_train, endurance_train)

    endurance_fitted = linearModel.predict(weight_train)

    k = linearModel.coef_[0][0]
    b = linearModel.intercept_[0]


def readBase():
    with open('./0-baseInfo.txt', 'r') as f:
        text = f.readlines()
    for line in text:
        info = line.split(',')
        x = int(info[1])
        y = int(info[2][:-2])
        baseList.append([x,y,UAVPerBase])
        # 初始化每个base里的无人机
        for uIndex in range(UAVPerBase):
            UAVNo = uIndex + UAVPerBase * (len(baseList) - 1)
            UAVInfo = [UAVNo, len(baseList) - 1, x, y, UAVEndurance, 1]
            UAVList.append(UAVInfo)
        if len(baseList) == baseNum:        # 读够需要的base后就停止
            break


# 统计属于某一个基站的无人机总数及空闲无人机数，一起返回
def getUAVNum(baseNo):
    # 因为UAVList每个slot都在更新，所以这里不需要传slot这个参数
    UAVinBase = 0
    FreeUAV = 0
    for UAV in UAVList:
        base = UAV[1]
        if base == baseNo:
            UAVinBase += 1
            if UAV[-1] == 1:
                FreeUAV += 1
    return UAVinBase, FreeUAV


def chooseTaxisAndReadTrajectory():
    # 插值工作已经在2-0-insertTaxiWP.py中完成，因此本文件直接读取插值完成的文件即可
    global taxiList, taxiNum
    # note taxi trajectory和passenger trip都是从day1开始的，这里全部+1
    taxiFileName = './2-taxiList/2-taxiList_' + str(taxiProportion) + '_day' + str(day+1) + '.txt'
    with open(taxiFileName,'r') as f:
        text = f.readlines()
    taxiNum = len(text)
    for line in text:
        taxiList.append(line.split('\n')[0])

    # (t1.latitude * 100) - 3070)
    # (t1.longitude * 100) - 12110))
    filePath = 'D:/0-DATA/Taxi_trajectory_data/Shanghai/' + str(day+1) + '/byCars_full/'
    taxiIndex = 0
    for taxi in taxiList:
        taxiTrajectoryList.append([])
        # read trips by taxi
        mycursor.execute("select * from uav_taxi_delivery.dbo.passenger_trips_day" + str(day+1) + " where taxiNo=?",taxi)
        for para in mycursor:
            pickTime = para[1]
            pickSlot = time_sub(pickTime, '08:00:00') // (slotLength * 60)
            dropTime = para[4]
            dropSlot = time_sub(dropTime, '08:00'
                                          ':00') // (slotLength * 60)
            pickLon = int(float(para[2]) * 100 - 12110)
            pickLat = int(float(para[3]) * 100 - 3070)
            dropLon = int(float(para[5]) * 100 - 12110)
            dropLat = int(float(para[6]) * 100 - 3070)

            if pickSlot < 0 or dropSlot >= slotNum:         # 开始于早八前，结束语晚八后的出租车订单不考虑
                continue
            # 不在地图内的trip也不考虑
            if 0 <= pickLon < 80 and 0 <= pickLat < 80 and 0 <= dropLon < 80 and 0 <= dropLat < 80:
                avgSpeedInTrip = distance_coefficient * cal_distance(pickLon,pickLat, dropLon,dropLat) / para[-1]      # 单位为km/min
                line = [para[0],pickSlot,pickLon,pickLat,dropSlot,dropLon,dropLat,avgSpeedInTrip,para[-1]]
                # taxiNo, pick time, pick lon, pick lat, drop time, drop lon, drop lat, avgSpeed, trip duration
                tripList[pickSlot].append(line)
                tripDepartureMap[pickSlot][pickLon][pickLat] += 1
                tripArrivalMap[dropSlot][dropLat][dropLon] += 1

        # note. read estimated trips
        mycursor.execute("select * from uav_taxi_delivery.dbo.estimated_passenger_trips_day" + str(day+1) + " where taxiNo=?",taxi)
        for para in mycursor:
            pickTime = para[1]
            pickSlot = time_sub(pickTime, '08:00:00') // (slotLength * 60)
            dropTime = para[4]
            dropSlot = time_sub(dropTime, '08:00'
                                          ':00') // (slotLength * 60)
            pickLon = int(float(para[2]) * 100 - 12110)
            pickLat = int(float(para[3]) * 100 - 3070)
            dropLon = int(float(para[5]) * 100 - 12110)
            dropLat = int(float(para[6]) * 100 - 3070)

            if pickSlot < 0 or dropSlot >= slotNum:         # 开始于早八前，结束语晚八后的出租车订单不考虑
                continue
            # 不在地图内的trip也不考虑
            if 0 <= pickLon < 80 and 0 <= pickLat < 80 and 0 <= dropLon < 80 and 0 <= dropLat < 80:
                avgSpeedInTrip = distance_coefficient * cal_distance(pickLon,pickLat, dropLon,dropLat) / para[-1]      # 单位为km/min
                line = [para[0],pickSlot,pickLon,pickLat,dropSlot,dropLon,dropLat,avgSpeedInTrip,para[-1]]
                # taxiNo, pick time, pick lon, pick lat, drop time, drop lon, drop lat, avgSpeed, trip duration
                estimated_tripList[pickSlot].append(line)

        # read trajectory by taxi
        fileName = filePath + '/' + str(taxi) + '.csv'
        slotOccupied = np.zeros((slotNum,),dtype=bool)        # 0表示还没有该时刻的轨迹点， 1表示已有
        with open(fileName, 'r') as f:
            text = f.readlines()
        for wp in text:
            info = wp.split(',')
            slot = int(info[2])

            lon = int(info[3])
            lat = int(info[4])
            speed = float(info[5])
            line = [info[0],info[1],slot,lon,lat,speed]     # taxi id, status (0:busy, 1:free), slot, lon, lat, speed
            taxiTrajectoryList[taxiIndex].append(line)
            slotOccupied[slot] = True
        taxiIndex += 1


def readDelivery():
    # actual delivery
    global deliveryNum
    actualDBName = 'takeout_day' + str(day+24)            # 实际的外卖是day24-day30共七天
    mycursor.execute("select count(*) as rows from uav_taxi_delivery.dbo." + actualDBName)
    # mycursor.execute("select count(*) as rows from uav_taxi_delivery.dbo.takeOutData_day2")
    for para in mycursor:
        allDelivery = int(para[0])
        # ("all delivery", allDelivery)
    deliveryNum = int(allDelivery * deliveryProportion)
    # print("by taxi and UAV", deliveryNum)
    mycursor.execute("select * from uav_taxi_delivery.dbo." + actualDBName)
    # mycursor.execute("select * from uav_taxi_delivery.dbo.takeOutData_day2")
    deliCount = 0
    for para in mycursor:
        startTime = time_sub(str(para[0]), '08:00:00') // (slotLength * 60)
        endTime = time_sub(str(para[1]), '08:00:00') // (slotLength * 60)
        if startTime >= slotNum or startTime < 0 or endTime >= slotNum or endTime < 0:
            continue
        startLoc = para[2]
        endLoc = para[3]
        # 数据库中寸的是ding yi的原始方法，即
        # ((FLOOR(t1.latitude*100)-3070)*80+(FLOOR(t1.longitude*100)-12110)) AS grid_id
        startLon = startLoc % 80
        startLat = startLoc // 80
        endLon = endLoc % 80
        endLat = endLoc // 80
        actualDeliveryList[startTime].append([startTime, endTime, startLon, startLat, endLon, endLat])
        deliCount += 1
        if deliCount >= deliveryNum:  # 只读一定比例的外卖，因为这个数据库内的数据本来就是乱的，所以不需要再打乱
            break

    # predicted delivery
    predictedDBName = 'prediction_takeout_day' + str(day)           # 预测时记录的是day0-day1，对应实际的day24-day30
    mycursor.execute("select * from uav_taxi_delivery.dbo." + predictedDBName)
    # mycursor.execute("select * from uav_taxi_delivery.dbo.takeOutData_day3")
    for para in mycursor:
        startTime = int(para[0])            # predicted的外卖时间直接保存的是slot，直接读取即可
        endTime = int(para[1])
        if startTime >= slotNum or startTime < 0 or endTime >= slotNum or endTime < 0:
            continue
        startLoc = para[2]
        endLoc = para[3]
        # 数据库中寸的是ding yi的原始方法，即
        # ((FLOOR(t1.latitude*100)-3070)*80+(FLOOR(t1.longitude*100)-12110)) AS grid_id
        startLon = startLoc % 80
        startLat = startLoc // 80
        endLon = endLoc % 80
        endLat = endLoc // 80
        predictedDeliveryList[startTime].append([startTime, endTime, startLon, startLat, endLon, endLat])


# 计算每个grid对基站的归属
def gridBelonging():
    global gridBelongingMap
    gridBelongingMap.fill(-1)               # 将grid归属设定为-1
    for x in range(80):
        for y in range(80):
            distToBaseList = []
            for bIndex in range(len(baseList)):
                bx = int(baseList[bIndex][0])
                by = int(baseList[bIndex][1])
                dist = cal_distance(x,y,bx,by)
                distToBaseList.append([dist, bIndex])
            distToBaseList.sort(key=lambda x: x[1])
            nearestBaseIndex = distToBaseList[0][1]
            gridBelongingMap[x][y] = nearestBaseIndex


def cooperativeDelivery():
    def UAVinBaseStatus():       # note 这个有意义吗？ 因为这只能统计当前slot的无人机情况，偶然误差很大
        # 统计当前时刻每个基站中无人机的空闲程度
        UAVNumInBase = np.zeros((baseNum,),dtype=int)       # 记录每个基站无人机的数量
        BusyUAVInBase = np.zeros((baseNum,),dtype=int)       # 记录每个基站外出无人机的数量
        for UAV in UAVList:
            base = UAV[1]
            status = UAV[5]
            UAVNumInBase[base] += 1
            if status == 0:
                BusyUAVInBase[base] += 1
        # print(BusyUAVInBase / UAVNumInBase)

    def UAVRepositioning(slot):
        global allRepositionedUAVNum
        entropyList = [[[] for y in range(80)] for x in range(80)]  # 记录每一个grid的entropy，30min的entropy放在同一个List中
        entropyMap = np.zeros((80, 80), dtype=float)  # 将entropy List中的数据计算均值，作为repositioning的依据
        tripTempList = [[] for t in range(30)]
        deliveryTempList = [[] for t in range(30)]  # 30为30min
        gridTripCapacityLack = np.zeros((80, 80, 30), dtype=int)  # 记录往后30分钟每分钟每个grid的运力缺口

        for deliSlot in range(30):
            #  用 predicted delivery list
            deliveryTempList[deliSlot] = predictedDeliveryList[deliSlot + slot]
            # 本函数内的变量范围是0-30，而读取的外卖数据则是0-720，因此在读取外部外卖数据时要+slot
        for tripSlot in range(30):
            # tripTempList[tripSlot] = tripList[tripSlot + slot]
            tripTempList[tripSlot] = estimated_tripList[tripSlot + slot]
        # 在UAV repositioning中，已经将未来30min的trip和delivery的时间都转化为了0-30，所以后面全用0-30
        for nowSlot in range(min(slotNum - slot, 30)):
            for delivery in deliveryTempList[nowSlot]:
                availableTripList_Re = []
                deliStartTime = nowSlot
                deliStartLon = delivery[2]
                deliStartLat = delivery[3]
                deliEndTime = delivery[1]
                deliEndLon = delivery[4]
                deliEndLat = delivery[5]
                for trip in tripTempList[nowSlot]:
                    taxiNo = trip[0]
                    pickLon = trip[2]
                    pickLat = trip[3]
                    if cal_distance(pickLon, pickLat, deliStartLon, deliStartLat) <= 1:  # 相邻的正方向4个grid都可接受
                        dropTime = trip[4]
                        dropLon = trip[5]
                        dropLat = trip[6]
                        deliEndToPick = distance_coefficient * cal_distance(deliEndLon, deliEndLat, pickLon, pickLat)
                        deliEndToDrop = distance_coefficient * cal_distance(deliEndLon, deliEndLat, dropLon, dropLat)
                        distTripDirect = distance_coefficient * cal_distance(pickLon, pickLat, dropLon, dropLat)
                        # 可能中途送达
                        if deliEndToDrop < distTripDirect and deliEndToPick < distTripDirect:
                            distToTripDirect = dis_point_to_seg_line([deliEndLon, deliEndLat], [pickLon, pickLat],
                                                                     [dropLon, dropLat])
                            if distToTripDirect <= distTripDirect / 2:
                                # 若外卖终点落在以trip起止点为斜边的正三角形范围内就简单地认为可以中途送达，
                                # 因为trip的轨迹应在trip起止点线段至以起止点为直角边的三角形之间
                                detourDistSimplified = (cal_distance(deliStartLon, deliStartLat, pickLon, pickLat)
                                                        + distToTripDirect)
                                availableTripList_Re.append(
                                    [detourDistSimplified, taxiNo, dropTime, dropLon, dropLat, nowSlot])
                        else:
                            # 认为有可能起止点送达，计算detour dist
                            detourDistSimplified = distance_coefficient * (cal_distance(deliStartLon, deliStartLat, pickLon, pickLat)
                                                    + cal_distance(deliEndLon, deliEndLat, dropLon, dropLat))
                            if detourDistSimplified <= detourTimeLimitSimplified * taxiSpeedDefault:
                                # 简单地认为可以送达
                                availableTripList_Re.append(
                                    [detourDistSimplified, taxiNo, dropTime, dropLon, dropLat, nowSlot])
                availableTripList_Re.sort(key=lambda x: x[0])  # 按绕路距离排序
                if len(availableTripList_Re) == 0:
                    gridTripCapacityLack[deliStartLon][deliStartLat][nowSlot] += 1  # 没有available trip的订单运力缺口直接+1
                    entropyList[deliStartLon][deliStartLat].append(0)  # 同时entropy加一个0
                else:
                    availableTrip = availableTripList_Re[0]
                    dropTime = availableTrip[2]
                    dropLon = availableTrip[3]
                    dropLat = availableTrip[4]
                    passNum = tripDepartureMap[dropTime][dropLon][dropLat]
                    freeTaxiNum = tripArrivalMap[dropTime][dropLon][dropLat]
                    detourDist = availableTrip[0]
                    if freeTaxiNum == 0:
                        if passNum > 0:
                            taxiLeisureProbability = 0
                        else:
                            taxiLeisureProbability = 1
                    else:
                        taxiLeisureProbability = 1 - passNum / freeTaxiNum  # 该出租车送达当前乘客后空车的概率
                    if taxiLeisureProbability <= taxiLeisureProbabilityMin:  # 利用空车概率最最后的判断，如概率小则缺口+1
                        # 因为在available trip的判断上已经卡过绕路距离了，这里再卡一下空车概率即可
                        gridTripCapacityLack[deliStartLon][deliStartLat][nowSlot] += 1
                    if detourDist == 0:
                        detourDist = 0.1        # 对于起终点都不绕路的情况，假定绕路1km以防止0作除数
                    entropy = taxiLeisureProbability / detourDist
                    entropyList[deliStartLon][deliStartLat].append(entropy)  # 将该订单的entropy存到list里
                    tripSlot = availableTrip[5]
                    taxiNo = availableTrip[1]
                    for tripIndex in range(len(tripTempList[tripSlot])):  # 通过时间和taxiNo唯一确定一个trip，然后删掉
                        if tripTempList[tripSlot][tripIndex][0] == taxiNo:
                            break
                    del tripTempList[tripSlot][tripIndex]  # 该trip有配送的包裹了，删除该订单
                    break
        # 计算各grid的entropy
        for x in range(80):
            for y in range(80):
                if len(entropyList[x][y]) > 0:
                    entropyMap[x][y] = sum(entropyList[x][y]) / len(entropyList[x][y])
        # 以上完成了各grid entropy的计算，下面计算K值，并进行聚类，从而确定repositionning的方向

        # 直接利用每个基站运力的缺口计算调度的方向
        baseUAVRemain = []  # 存放每个base对UAV需求的数量，可正可负，正表示缺n架，负表示余-n架，数值为30个slot里的最大值
        baseTripCapacityLack = np.zeros((baseNum, 30), dtype=int)  # 每个base的trip运力缺口
        for baseIndex in range(baseNum):
            UAVRemainList = []  # 记录该base在30个slot内UAV Remain的平均值作为调度的指标
            # ↑使用平均值是为了避免某基站某一时刻突然出现大量外卖导致的大量无人机的提前调度
            baseUAVNum, freeUAVNum = getUAVNum(baseIndex)  # 分别对应该base UAV的总数以及开始repositioning时该基站空闲的无人机数
            for nowSlot in range(min(slotNum - slot, 30)):
                for x in range(80):
                    for y in range(80):
                        belongTo = gridBelongingMap[x][y]
                        # 计算每个基站在任意时刻的trip运力缺口，下面的就是要由无人机配送的包裹数
                        baseTripCapacityLack[belongTo][nowSlot] += gridTripCapacityLack[x][y][nowSlot]
                # 将UAV的数量与baseTripCapacityLack数值比较，确定UAV数量缺口
                # 假定一架无人机配送一个外卖从接到订单到返回基站要5分钟，则当前时刻基站的空闲无人机数 = 基站无人机总数 - sum(前五分钟的trip lack数)
                dispatchedUAVNum = 0
                for deliSlot in (max(0, nowSlot - 5), nowSlot):  # 这个deliSlot指 发生delivery的slot
                    dispatchedUAVNum += baseTripCapacityLack[baseIndex][deliSlot]
                if nowSlot <= 3:  # 认为在开始repositioning时飞走的无人机将在3分钟后回来
                    UAVRemainSlot = freeUAVNum - dispatchedUAVNum - baseTripCapacityLack[baseIndex][nowSlot]
                else:  # 3分钟后认为无人机都回来了，直接使用UAV in base计算lack
                    UAVRemainSlot = baseUAVNum - dispatchedUAVNum - baseTripCapacityLack[baseIndex][nowSlot]
                UAVRemainList.append(UAVRemainSlot)
            avgUAVRemain = int(sum(UAVRemainList) / len(UAVRemainList))  # 向下取整
            baseUAVRemain.append([baseIndex, avgUAVRemain])
        baseUAVRemain.sort(key=lambda x: x[1])  # 用avg uav remain升序排序
        for baseNumIn in range(len(baseUAVRemain)):
            repositioningComplete = 0  # 标记是否完成了该base in的所有调度
            baseIndexIn = baseUAVRemain[baseNumIn][0]
            UAVRemainIn = baseUAVRemain[baseNumIn][1]
            if UAVRemainIn >= 0:  # 升序排序的情况下遇到了不缺UAV的base则证明repositioning已经完成
                break
            for baseNumOut in range(baseNumIn, len(baseUAVRemain)):
                baseIndexOut = baseUAVRemain[baseNumOut][0]
                UAVRemainOut = baseUAVRemain[baseNumOut][1]
                if UAVRemainOut > 0:  # 这个base有空余的无人机，则进行调度
                    if abs(UAVRemainIn) <= UAVRemainOut:
                        # 供过于求，则调度需求的
                        repositioningNum = abs(UAVRemainIn)
                    else:
                        # 供不应求，则调度能供给的
                        repositioningNum = abs(UAVRemainOut)
                    # 修改UAVList的数据，标记调度的无人机为busy
                    uIndex = -1
                    repositionedNum = 0  # 已重新调度的无人机数
                    for UAV in UAVList:
                        uIndex += 1
                        baseNo = UAV[1]
                        if baseNo == baseIndexOut:  # 该UAV属于调出UAV的base
                            baseLonIn = baseList[baseIndexIn][0]
                            baseLatIn = baseList[baseIndexIn][1]
                            baseLonOut = baseList[baseIndexOut][0]
                            baseLatOut = baseList[baseIndexOut][1]
                            slotToBaseIn = int(
                                cal_distance(baseLonIn, baseLatIn, baseLonOut, baseLatOut) / UAVSpeed) + 1  # 到达时间向上取正
                            UAVInfo = [UAV[0]] + [baseIndexIn] + UAV[2:] + [-1, slot + slotToBaseIn, -1, -1]
                            UAVList[uIndex] = UAVInfo
                            allRepositionedUAVNum += 1
                            repositionedNum += 1
                        if repositionedNum == repositioningNum:
                            # 已经完成了base in 所有需要调度的无人机的调度
                            repositioningComplete = 1
                            break
                if repositioningComplete == 1:
                    break

        # determine the origin and destination of UAV repositioning

        # dispatching number

    def updateUAVStatus(slot):
        uIndex = -1
        for UAV in UAVList:
            uIndex += 1
            if len(UAV) > 6:  # 正在配送外卖或返回基站中；len为6的就是在基站的无人机，不需要维护
                deliveryEndSlot = UAV[6]
                status = UAV[5]
                if slot >= deliveryEndSlot and status == 0:
                    # 配送已完成，需要将无人机的status更改为free(1)
                    UAV[5] = 1  # 当status为1时无论UAV[6]为何值都不管
                elif status == 1:
                    baseNo = UAV[1]
                    baseX = baseList[baseNo][0]
                    baseY = baseList[baseNo][1]
                    if slot >= UAV[7]:  # 当前时刻超过返回基站的时刻，则无人机返回基站
                        UAVList[uIndex] = UAV[:2] + [baseX, baseY, UAVEndurance, 1]
                    else:
                        # status为1但len仍大于5表明无人机正在返回基站，这时需要修改位置及剩余能量
                        deliEndLon = UAV[8]
                        deliEndLat = UAV[9]
                        totalBackTime = UAV[7] - UAV[6]
                        passedTime = slot - UAV[6]
                        newX = (baseX - deliEndLon) / totalBackTime * passedTime + deliEndLon
                        newY = (baseY - deliEndLat) / totalBackTime * passedTime + deliEndLat
                        newEnergy = UAV[4] - 1  # 每个slot都更新，因此直接-1就行
                        UAVList[uIndex] = UAV[:2] + [newX, newY, newEnergy] + UAV[5:]

    global allRepositionedUAVNum, freeTaxiDeliveredNum
    for slot in range(slotNum):  # 每个slot 1min
        if slot % 30 == 0 and slot > 0:
            UAVRepositioning(slot)
            # print("now is", slot, allRepositionedUAVNum)
            UAVinBaseStatus()
        updateUAVStatus(slot)

        for delivery in actualDeliveryList[slot]:
            deliStartTime = slot
            deliStartLon = delivery[2]
            deliStartLat = delivery[3]
            deliEndTime = delivery[1]
            deliEndLon = delivery[4]
            deliEndLat = delivery[5]

            availableTripList = []  # 存放可以用于配送该包裹的所有trip数据
            for tripSlot in range(max(0, slot - 1), min(slotNum, slot + PTL)):
                # note. Temporal Match: 发生delivery前1分钟与后5分钟的所有订单
                # ↑基于认知：网约车都提前3-5分钟下单
                for trip in tripList[tripSlot]:
                    taxiNo = trip[0]
                    taxiPickLon = trip[2]
                    taxiPickLat = trip[3]
                    taxiDropSlot = int(trip[4])
                    taxiDropLon = trip[5]
                    taxiDropLat = trip[6]
                    # avgSpeed = trip[-2]
                    # if avgSpeed == 0:
                    avgSpeed = taxiSpeedDefault
                    taxiIndex = taxiList.index(taxiNo)  # 从taxiList中找出该taxi对应的index
                    if len(taxiTrajectoryList[taxiIndex][slot]) == 7:
                        # 原本应该是6位，当被占用时会增加一位flag表示该出租车正配送包裹
                        continue
                    taxiTraj = taxiTrajectoryList[taxiIndex]  # 利用taxi index提取该taxi的trajectory
                    taxiLon = taxiTraj[slot][3]
                    taxiLat = taxiTraj[slot][4]
                    detourOrigin = distance_coefficient * (cal_distance(deliStartLon, deliStartLat, taxiPickLon, taxiPickLat)
                                    + cal_distance(deliStartLon, deliStartLat, taxiLon, taxiLat)
                                    - cal_distance(taxiPickLon, taxiPickLat, taxiLon, taxiLat))
                    if detourOrigin < 0:
                        detourOrigin = 0
                    detourTimeOrigin = detourOrigin / avgSpeed

                    # 判断中途送达的可能
                    modeC = 0
                    if detourTimeOrigin <= originTime_in:
                        dropTime = trip[4]
                        wpInTrip = taxiTraj[tripSlot:dropTime + 1]
                        for wpIndex in range(len(wpInTrip)):
                            wpLon = wpInTrip[wpIndex][3]
                            wpLat = wpInTrip[wpIndex][4]
                            # 中途配送允许绕路距离
                            if distance_coefficient * cal_distance(wpLon, wpLat, deliEndLon, deliEndLat) <= destTime_in * avgSpeed:
                                wpTime = wpInTrip[wpIndex][2]
                                deliTimeDest = distance_coefficient * cal_distance(wpLon, wpLat, deliEndLon,
                                                            deliEndLat) / avgSpeed  # 送达外卖的单程时间
                                deliveryTime = detourTimeOrigin + wpTime - tripSlot + deliTimeDest  # 起点绕路 + 半程trip + 终点绕路
                                detourTimeDest = deliTimeDest * 2  # 绕路时间为往返
                                detourTimeSum = detourTimeOrigin + detourTimeDest
                                line = [taxiNo, tripSlot, detourTimeSum, deliveryTime, 1, detourTimeOrigin]  # 0 表示目的地送达； 1表示中途送达
                                availableTripList.append(line)
                                modeC = 1
                                break
                    if modeC == 1:  # 该trip可以中途送达该包裹，不进行该trip终点匹配的计算
                        break

                    # 判断是否有可能 起点终点送达
                    deliDestTime = distance_coefficient * cal_distance(deliEndLon, deliEndLat, taxiDropLon,
                                                taxiDropLat) / avgSpeed  # 出租车将外卖包裹送到的时间， 单位为km / (km/min) = min

                    if int(deliDestTime + taxiDropSlot) + 1 >= 720:  # 出租车不送外卖就要晚8以后才送达，这种trip不送包裹
                        continue
                    wpNoDeli = taxiTraj[int(deliDestTime + taxiDropSlot) + 1]  # 若不送包裹，相同时间出租车会到这里
                    wpNoDeliLon = wpNoDeli[3]
                    wpNoDeliLat = wpNoDeli[4]
                    detourDest = distance_coefficient * cal_distance(wpNoDeliLon, wpNoDeliLat, deliEndLon, deliEndLat)  # 送达外卖后出租车直接返回本应出现的点
                    detourTimeDest = detourDest / avgSpeed
                    detourTimeSum = detourTimeOrigin + detourTimeDest
                    if detourTimeOrigin <= originTime_OD and detourDest <= destTime_OD:  # 起点终点送达
                        # 用if是因为有的订单可能会进上面的if，但实际上应由起点终点送达。
                        # 起点终点送达放下面是因为中途送达更快，因此判断的优先级更高
                        deliveryTime = detourTimeOrigin + deliDestTime + taxiDropSlot - tripSlot  # 起点绕路 + 终点绕路 + trip时长 -> delivery time
                        # 0 表示目的地送达； 1表示中途送达
                        line = [taxiNo, tripSlot, detourTimeSum, deliveryTime, 0, detourTimeOrigin]
                        availableTripList.append(line)
            # 以上计算了将所有的available trip
            if len(availableTripList) > 0:
                # 存在可以送这单外卖的trip
                availableTripList.sort(key=lambda x: x[2])  # detour time sum升序排列
                bestTrip = availableTripList[0]
            else:
                bestTrip = []  # 如果没有trip可以送外卖，则best trip为空
            availableUAVList = []
            # 查询无人机的空闲程度
            for UAV in UAVList:
                if UAV[5] == 1:
                    # 无人机空闲，计算该架次无人机配送这个包裹需要的总能量
                    UAVX = UAV[2]
                    UAVY = UAV[3]
                    baseX = baseList[UAV[1]][0]
                    baseY = baseList[UAV[1]][2]
                    distSum = (cal_distance(UAVX, UAVY, deliStartLon, deliStartLat)
                               + cal_distance(deliStartLon, deliStartLat, deliEndLon, deliEndLat)
                               + cal_distance(deliEndLon, deliEndLat, baseX, baseY))
                    deliveryWeight = random.uniform(0.4,0.8)           # 单位为kg: https://zhuanlan.zhihu.com/p/34627013
                    timeConsumption = distSum / UAVSpeed
                    energyConsumptionCoefficient = UAVEndurance / (deliveryWeight*k+b)      # 搭载该包裹对能耗的影响，如1.3倍等
                    if timeConsumption * energyConsumptionCoefficient <= UAV[4]:  # 无人机的剩余电量足够完成本次配送
                        deliveryTime = (cal_distance(UAVX, UAVY, deliStartLon, deliStartLat)
                                        + cal_distance(deliStartLon, deliStartLat, deliEndLon, deliEndLat)) / UAVSpeed
                        UAVInfo = [UAV[0], deliveryTime, timeConsumption]
                        availableUAVList.append(UAVInfo)
            availableUAVList.sort(key=lambda x: x[1])  # 按delivery time升序排序
            if len(availableUAVList) > 0:
                bestUAV = availableUAVList[0]
            else:
                bestUAV = []

            if len(bestTrip) > 0 and len(bestUAV) == 0:  # 仅有出租车能配送
                deliveryTime = bestTrip[3]
                formerDeliveryTime = deliEndTime - deliStartTime
                taxiNo = bestTrip[0]
                taxiIndex = taxiList.index(taxiNo)
                deliveryPack = [deliveryTime, 0, formerDeliveryTime,deliStartTime,taxiIndex]
                deliveredPackageList.append(deliveryPack)
                taxiDetourList.append(bestTrip[2])          # best trip[2]为detourTime, detourOrigin + detourEnd
                detourTimeOrigin = bestTrip[5]
                if detourTimeOrigin < 0:
                    continue
                detourTimeOriginList.append(detourTimeOrigin)
                for trajSlot in range(deliStartTime, min(int(deliStartTime + deliveryTime) + 1,slotNum)):
                    taxiTrajectoryList[taxiIndex][trajSlot] += [-1]  # 给出租车轨迹新增一位-1表示该车正在配送包裹
            elif len(bestTrip) == 0 and len(bestUAV) > 0:  # 仅有无人机能配送
                deliveryTime = bestUAV[1]
                timeConsumption = bestUAV[2]
                formerDeliveryTime = deliEndTime - deliStartTime
                UAVIndex = bestUAV[0]
                deliveryPack = [deliveryTime, 1, formerDeliveryTime,deliStartTime,UAVIndex]
                deliveredPackageList.append(deliveryPack)
                UAVNo = bestUAV[0]  # UAVNo 在数值上等于UAV index
                energy = UAVList[UAVNo][4]
                energyAfterDelivery = energy - deliveryTime * energyConsumptionCoefficient   # 对无人机的能耗加一个系数
                # 将UAV标记为busy，将位置修改为外卖终点的位置 并在最后加上下次free 和 返回基站的时间
                UAVList[UAVNo] = UAVList[UAVNo][:2] + [deliEndLon, deliEndLat, energyAfterDelivery, 0,
                                                       deliStartTime + deliveryTime, deliStartTime + timeConsumption,
                                                       deliEndLon, deliEndLat]
            elif len(bestTrip) > 0 and len(bestUAV) > 0:  # 既有无人机也有出租车能配送
                taxiDetourTime = bestTrip[2]
                UAVTimeConsumption = bestUAV[2]
                if taxiDetourTime <= 1.3 * UAVTimeConsumption:  # 1.5   -> 1.3
                    # 当出租车仅用少于1.5倍无人机耗时的情况下就可以完成包裹的配送，就让出租车配送
                    deliveryTime = bestTrip[3]
                    formerDeliveryTime = deliEndTime - deliStartTime
                    taxiNo = bestTrip[0]
                    taxiIndex = taxiList.index(taxiNo)
                    deliveryPack = [deliveryTime, 0, formerDeliveryTime, deliStartTime, taxiIndex]
                    deliveredPackageList.append(deliveryPack)
                    taxiDetourList.append(bestTrip[2])
                    detourTimeOrigin = bestTrip[5]
                    if detourTimeOrigin < 0:
                        continue
                    detourTimeOriginList.append(detourTimeOrigin)
                    for trajSlot in range(deliStartTime, min(int(deliStartTime + deliveryTime) + 1, slotNum)):
                        taxiTrajectoryList[taxiIndex][trajSlot] += [-1]  # 给出租车轨迹新增一位-1表示该车正在配送包裹
                else:
                    # 否则，由无人机配送
                    deliveryTime = bestUAV[1]
                    timeConsumption = bestUAV[2]
                    formerDeliveryTime = deliEndTime - deliStartTime
                    UAVNo = bestUAV[0]  # UAVNo 在数值上等于UAV index
                    deliveryPack = [deliveryTime, 1, formerDeliveryTime,deliStartTime,UAVNo]
                    deliveredPackageList.append(deliveryPack)
                    energy = UAVList[UAVNo][4]
                    energyAfterDelivery = energy - deliveryTime * energyConsumptionCoefficient  # 对无人机的能耗加一个系数
                    # 将UAV标记为busy，将位置修改为外卖终点的位置 并在最后加上下次free 和 返回基站的时间
                    UAVList[UAVNo] = UAVList[UAVNo][:2] + [deliEndLon, deliEndLat, energyAfterDelivery, 0,
                                                           deliStartTime + deliveryTime,
                                                           deliStartTime + timeConsumption,
                                                           deliEndLon, deliEndLat]
            else:
                # 既没有出租车配送，也没有无人机配送，则判断是否有空车可以配送
                if allowFreeDelivery == 0:          # 不允许空车配送时直接continue
                    undeliveredPackageList.append(delivery)
                    continue
                freeTaxiDeliver = 0  # 标记是否由空车配送
                for taxiIndex in range(taxiNum):
                    taxiX = taxiTrajectoryList[taxiIndex][deliStartTime][3]
                    taxiY = taxiTrajectoryList[taxiIndex][deliStartTime][4]
                    onTrip = taxiTrajectoryList[taxiIndex][deliStartTime][2]
                    if (len(taxiTrajectoryList[taxiIndex][deliStartTime]) > 6 or onTrip == 0
                            or distance_coefficient * cal_distance(taxiX, taxiY, deliStartLon, deliStartLat) > originTime_OD * taxiSpeedDefault):
                        # 出租车正在送外卖 或 乘客 或 出租车此时的距离距外卖的开始点超过2公里
                        continue
                    taxiSpeed = taxiSpeedDefault
                    totalDist = distance_coefficient * (cal_distance(taxiX, taxiY, deliStartLon, deliStartLat)
                                 + cal_distance(deliStartLon, deliStartLat, deliEndLon, deliEndLat))
                    deliveryTime = totalDist / taxiSpeed
                    taxiPickLon = -1            # initialize the pickup location to indict whether detoured or not
                    for deliverySlot in range(deliStartTime, min(int(deliStartTime + deliveryTime) + 1,slotNum)):
                        if taxiTrajectoryList[taxiIndex][deliStartTime][2] == 0:
                            taxiPickLon = int(taxiTrajectoryList[taxiIndex][deliStartTime][3])
                            taxiPickLat = int(taxiTrajectoryList[taxiIndex][deliStartTime][4])
                            taxiSpeed = float(taxiTrajectoryList[taxiIndex][deliStartTime][5])
                            # 在送外卖的途中出现了原应由该出租车送的乘客，则计算绕路距离
                            distEndToPass = distance_coefficient * cal_distance(deliEndLon,deliEndLat,taxiPickLon,taxiPickLat)      # 空车送完外卖后返回乘客起点的路程
                            detourTimeOrigin = deliveryTime - (deliverySlot - deliStartTime) + distEndToPass / taxiSpeed # 从出现乘客起到出租车送完该外卖的时间即为出租车的绕路时长
                            # TODO 检查detour time的计算，会出现负数？？
                            if detourTimeOrigin < 0:
                                continue
                            detourTimeOriginList.append(detourTimeOrigin)
                            for trajSlot in range(deliStartTime, min(int(deliStartTime + deliveryTime) + 1, slotNum)):
                                taxiTrajectoryList[taxiIndex][trajSlot] += [-1]  # 给出租车轨迹新增一位-1表示该车正在配送包裹
                            break
                    formerDeliveryTime = deliEndTime - deliStartTime
                    # 如果空车配送超过晚8的也不行就在这里加一个if-break即可
                    deliveryPack = [deliveryTime, 0, formerDeliveryTime,deliStartTime,taxiIndex]
                    deliveredPackageList.append(deliveryPack)
                    for trajSlot in range(deliStartTime, min(int(deliStartTime + deliveryTime) + 1, 720)):
                        # 此处表示空车配送，因此就算超过晚8了也行
                        taxiTrajectoryList[taxiIndex][trajSlot] += [-1]  # 给出租车轨迹新增一位-1表示该车正在配送包裹
                    freeTaxiDeliver = 1
                    freeTaxiDeliveredNum += 1
                    break
                if freeTaxiDeliver == 0:
                    # 没有空车可以配送，则标记该包裹为不可配送
                    undeliveredPackageList.append(delivery)


def getResults():

    def getParameter():
        if parameterChanged == 'deliveryProportion':
            return deliveryProportion
        elif parameterChanged == 'taxiProportion':
            return taxiProportion
        elif parameterChanged == 'UAVPerBase':
            return UAVPerBase
        elif parameterChanged == 'BaseNum':
            return baseNum
        elif parameterChanged == 'taxiLeisure':
            return taxiLeisureProbabilityMin
        elif parameterChanged == 'UAVEndurance':
            return UAVEndurance
        elif parameterChanged == 'UAVSpeed':
            return SecSpeed
        elif parameterChanged == 'detourTime':
            return detourTimeLimit
        elif parameterChanged == 'PTL':
            return PTL

    def calDestWaitingTime():
        deliListTaxi = [[] for taxi in range(taxiNum)]  # 将delivered package 按配送包裹的taxi分
        for deli in deliveredPackageList:
            if deli[1] == 0:  # 是出租车配送的
                taxiIndex = deli[4]
                deliListTaxi[taxiIndex].append(deli)
        slotTaxiTripMap = np.zeros((taxiNum, slotNum), dtype=bool)  # 记录每个taxi在不同slot的trip情况，1为在当前slot有trip,0为无
        for slot in range(slotNum):
            for trip in tripList[slot]:
                taxiNo = trip[0]
                taxiIndex = taxiList.index(taxiNo)
                slotTaxiTripMap[taxiIndex][slot] = True
        detourTimeDestList = []
        deliveredPackageList.sort(key=lambda x: x[1])  # 让出租车送的出现在最前面
        matchedPackList = []  # 记录已经计算过detour destination的package
        for slot in range(slotNum):
            for trip in tripList[slot]:
                taxiNo = trip[0]
                taxiIndex = taxiList.index(taxiNo)
                for pack in deliListTaxi[taxiIndex]:
                    if [pack[4], pack[3]] in matchedPackList:
                        continue
                    if max(0, slot - 5) < pack[3] < slot:  # 该包裹在这趟trip发生前5分钟发生
                        for trajSlot in range(slot, min(slotNum, slot + 5)):  # 统计往后5分钟的，因为允许dest绕路的最长时间也不超过5分钟
                            if len(taxiTrajectoryList[taxiIndex][trajSlot]) == 6:
                                # print(taxiIndex,slot)
                                detourTimeDest = trajSlot - slot  # 如果没绕路则trajSlot == slot，因此不需要再加if分支
                                detourTimeDestList.append(detourTimeDest)
                                matchedPackList.append([taxiIndex, pack[3]])  # 通过taxi index以及package slot唯一确定一个pack
                                break
        return detourTimeDestList

    detourTimeDestList = calDestWaitingTime()
    # print('~~~~~~~~~~~~~~~~~RESULTS~~~~~~~~~~~~~~~~~~~~~')
    # print('avg extra waiting time origin', sum(detourTimeOriginList) / len(detourTimeOriginList))
    # print('avg extra waiting time destination', sum(detourTimeDestList) / len(detourTimeDestList))
    # print('avg taxi detour distance', sum(taxiDetourList) / len(taxiDetourList))
    # print('free taxi delivery', freeTaxiDeliveredNum)
    # print('undelivered packages', len(undeliveredPackageList))
    # print('delivered packages', len(deliveredPackageList))
    byTaxi = 0
    byUAV = 0
    for pack in deliveredPackageList:
        if pack[1] == 0:
            byTaxi += 1
        else:
            byUAV += 1
    # print('by taxi', byTaxi, 'by UAV', byUAV)
    deliveryTimeList = []
    deliveryTimeFormerList = []
    for pack in deliveredPackageList:
        deliveryTimeList.append(pack[0])
        deliveryTimeFormerList.append(pack[2])
    # print('avg delivery time (min)', sum(deliveryTimeList) / len(deliveryTimeList))
    # print('avg delivery time by carriers (min)', sum(deliveryTimeFormerList) / len(deliveryTimeFormerList))

    # 统计空车的趟数，从而确定空车的使用比例
    freeCount = 0          # 空车的趟数
    for tIndex in range(taxiNum):
        formerStatus = 1
        if taxiTrajectoryList[tIndex][0][1] == 0:       # 该车的轨迹是从空车开始的，直接先加1
            freeCount += 1
        for wp in taxiTrajectoryList[tIndex]:
            status = int(wp[1])
            if status - formerStatus == -1:     # 空车是0，载客是1， minus == -1则为统计空车的开始，而非空车的结束
                freeCount += 1
            formerStatus = status
    # print (freeCount)

    path = './2-results/cooperation/'
    os.makedirs(path, exist_ok=True)         # note. 若path存在则无操作，若path不存在则创建该path
    resultFile = path + parameterChanged + '_day' + str(day) + '.txt'

    paraValue = getParameter()

    with open(resultFile, 'a') as f:
        line = 'free delivery ' + str(allowFreeDelivery) + ' ' + parameterChanged + ' ' + str(paraValue)
        results = (' delivery_time ' + str(sum(deliveryTimeList) / len(deliveryTimeList)) + ' delivered_packages '
                   + str(len(deliveredPackageList)) + ' undelivered_packages ' + str(len(undeliveredPackageList))
                   + ' extra_waiting_time ' + str(sum(detourTimeOriginList) / len(detourTimeOriginList)
                                                  + sum(detourTimeDestList) / len(detourTimeDestList))
                   + ' by_UAV ' + str(byUAV) + ' by_taxi ' + str(byTaxi) + ' free_occupation_ratio '
                   + str(freeTaxiDeliveredNum / freeCount) + ' free_count ' + str(freeCount))
        outputLine = line + results
        print(outputLine, file=f)
        detourListFile = path + parameterChanged + '_' + str(paraValue) + '_detourList' + '_day' + str(day) + '.txt'
        with open(detourListFile, 'a') as f:
            print('origin', detourTimeOriginList, file=f)
            # print('dest', detourTimeDestList, file=f)


if __name__ == '__main__':
    test_db = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};'
                             'SERVER=localhost;DATABASE=uav_taxi_delivery;UID=sa;PWD=gaoxiaojun53')
    test_db.autocommit = True
    mycursor = test_db.cursor()

    taxiTrajectoryList = []  # 已提前在2-0-*.py中插过值，因此这里直接读取的就是插值后的版本

    startAt = datetime.datetime.now()
    # print('execution starts at', startAt)
    weightLinearFit()           # 先拟合包裹重量对无人机续航的影响
    readBase()
    chooseTaxisAndReadTrajectory()
    readDelivery()
    gridBelonging()
    cooperativeDelivery()
    getResults()


    # print("execution time is", datetime.datetime.now() - startAt)
