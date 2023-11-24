######### This is a test file##############


import os
import datetime

deliveryProportion_default = 1      # uav+taxi配送包裹的比例
allowFreeDelivery_default = 1       # 是否允许空车配送, 0不允许，1允许
taxiProportion_default = 0.1        # 选中出租车的比例
UAVPerBase_default = 30           # 每个基站起始的无人机数  20
baseNum_default = 15
taxiLeisureProbabilityMin_default = 0.85         # 在uav repositioning时设定最小的空车概率为0.7，若空车概率小于0.7则运力缺口+1
UAVEndurance_default = 40           # 设定无人机续航40分钟
SecSpeed_default = 16
detourTimeLimit_default = 10         # 该数值==originTime_OD+destTime_OD，详情见2-cooperation.py
PTL_default = 5             # the default value of PTL is 5 min

parameterList = ['deliveryProportion','taxiProportion','UAVPerBase','BaseNum','taxiLeisure','UAVEndurance','UAVSpeed','detourTime']


def delResultsFiles(parameter,day):
    resultFile_cooperation = './2-results/cooperation/' + parameter + '_day' + str(day) + '.txt'
    resultFile_onlyTaxi = './2-results/onlyTaxi/' + parameter + '_day' + str(day) + '.txt'
    resultFile_onlyUAV = './2-results/onlyUAV/' + parameter + '_day' + str(day) + '.txt'
    os.remove(resultFile_cooperation)
    os.remove(resultFile_onlyTaxi)
    os.remove(resultFile_onlyUAV)


def calculateResults(day):
    #     os.system('python 2-cooperation.py %s %i %f %f %i %i %f %i %i %f %i %i' % (parameterChanged, allowFreeDelivery_default,
    #                  deliveryProportion,taxiProportion_default, UAVPerBase_default,baseNum_default,
    #                  taxiLeisureProbabilityMin_default,UAVEndurance_default,SecSpeed_default,detourTimeLimit_default,PTL_default,day))
    #     os.system('python 2-cooperation.py %s %i %f %f %i %i %f %i %i %f %i %i' % (parameterChanged, 0,
    #                  deliveryProportion,taxiProportion_default, UAVPerBase_default,baseNum_default,
    #                  taxiLeisureProbabilityMin_default,UAVEndurance_default,SecSpeed_default,detourTimeLimit_default,PTL_default,day))
    #     os.system('python 2-onlyTaxi.py %s %i %f %f %i %i %f %i %i %f %i %i' % (parameterChanged, allowFreeDelivery_default,
    #                  deliveryProportion,taxiProportion_default, UAVPerBase_default,baseNum_default,
    #                  taxiLeisureProbabilityMin_default,UAVEndurance_default,SecSpeed_default,detourTimeLimit_default,PTL_default,day))
    #     os.system('python 2-onlyTaxi.py %s %i %f %f %i %i %f %i %i %f %i %i' % (parameterChanged, 0,
    #                  deliveryProportion,taxiProportion_default, UAVPerBase_default,baseNum_default,
    #                  taxiLeisureProbabilityMin_default,UAVEndurance_default,SecSpeed_default,detourTimeLimit_default,PTL_default,day))
    #     # note 因为修改了delivery proportion的范围，因此这个参数的only UAV需要重新跑一下
    #     os.system('python 2-onlyUAV.py %s %i %f %f %i %i %f %i %i %f %i %i' % (parameterChanged, allowFreeDelivery_default,
    #                  deliveryProportion,taxiProportion_default, UAVPerBase_default,baseNum_default,
    #                  taxiLeisureProbabilityMin_default,UAVEndurance_default,SecSpeed_default,detourTimeLimit_default,PTL_default,day))
    # print('delivery proportion costs', datetime.datetime.now() - functionBegins)
    #
    # for taxiProportion in [0.05,0.1,0.15,0.2]:
    #     parameterChanged = 'taxiProportion'
    #     # 清空文件后再运行对应代码
    #     # delResultsFiles(parameterChanged,day)
    #     os.system('python 2-cooperation.py %s %i %f %f %i %i %f %i %i %f %i %i' % (parameterChanged, allowFreeDelivery_default,
    #                  deliveryProportion_default,taxiProportion, UAVPerBase_default,baseNum_default,
    #                  taxiLeisureProbabilityMin_default,UAVEndurance_default,SecSpeed_default,detourTimeLimit_default,PTL_default,day))
    #     os.system('python 2-cooperation.py %s %i %f %f %i %i %f %i %i %f %i %i' % (parameterChanged, 0,
    #                  deliveryProportion_default,taxiProportion, UAVPerBase_default,baseNum_default,
    #                  taxiLeisureProbabilityMin_default,UAVEndurance_default,SecSpeed_default,detourTimeLimit_default,PTL_default,day))
    #     os.system('python 2-onlyTaxi.py %s %i %f %f %i %i %f %i %i %f %i %i' % (parameterChanged, allowFreeDelivery_default,
    #                  deliveryProportion_default,taxiProportion, UAVPerBase_default,baseNum_default,
    #                  taxiLeisureProbabilityMin_default,UAVEndurance_default,SecSpeed_default,detourTimeLimit_default,PTL_default,day))
    #     os.system('python 2-onlyTaxi.py %s %i %f %f %i %i %f %i %i %f %i %i' % (parameterChanged, 0,
    #                  deliveryProportion_default,taxiProportion, UAVPerBase_default,baseNum_default,
    #                  taxiLeisureProbabilityMin_default,UAVEndurance_default,SecSpeed_default,detourTimeLimit_default,PTL_default,day))
    # print('taxi proportion costs', datetime.datetime.now() - functionBegins)
    #
    # for UAVPerBase in [20,25,30,35,40]:
    #     parameterChanged = 'UAVPerBase'
    #     # 清空文件后再运行对应代码
    #     # delResultsFiles(parameterChanged,day)
    #     os.system('python 2-cooperation.py %s %i %f %f %i %i %f %i %i %f %i %i' % (parameterChanged, allowFreeDelivery_default,
    #                  deliveryProportion_default,taxiProportion_default, UAVPerBase,baseNum_default,
    #                  taxiLeisureProbabilityMin_default,UAVEndurance_default,SecSpeed_default,detourTimeLimit_default,PTL_default,day))
    #     os.system('python 2-cooperation.py %s %i %f %f %i %i %f %i %i %f %i %i' % (parameterChanged, 0,
    #                  deliveryProportion_default,taxiProportion_default, UAVPerBase,baseNum_default,
    #                  taxiLeisureProbabilityMin_default,UAVEndurance_default,SecSpeed_default,detourTimeLimit_default,PTL_default,day))
    #     # os.system('python 2-onlyUAV.py %s %i %f %f %i %i %f %i %i %f %i %i' % (parameterChanged, allowFreeDelivery_default,
    #     #              deliveryProportion_default,taxiProportion_default, UAVPerBase,baseNum_default,
    #     #              taxiLeisureProbabilityMin_default,UAVEndurance_default,SecSpeed_default,detourTimeLimit_default,PTL_default,day))
    # print('UAV per base costs', datetime.datetime.now() - functionBegins)
    #
    # for BaseNum in [5,10,15,20,25]:
    #     parameterChanged = 'BaseNum'
    #     # 清空文件后再运行对应代码
    #     # delResultsFiles(parameterChanged,day)
    #     os.system('python 2-cooperation.py %s %i %f %f %i %i %f %i %i %f %i %i' % (parameterChanged, allowFreeDelivery_default,
    #                  deliveryProportion_default,taxiProportion_default, UAVPerBase_default,BaseNum,
    #                  taxiLeisureProbabilityMin_default,UAVEndurance_default,SecSpeed_default,detourTimeLimit_default,PTL_default,day))
    #     os.system('python 2-cooperation.py %s %i %f %f %i %i %f %i %i %f %i %i' % (parameterChanged, 0,
    #                  deliveryProportion_default,taxiProportion_default, UAVPerBase_default,BaseNum,
    #                  taxiLeisureProbabilityMin_default,UAVEndurance_default,SecSpeed_default,detourTimeLimit_default,PTL_default,day))
    #     # os.system('python 2-onlyUAV.py %s %i %f %f %i %i %f %i %i %f %i %i' % (parameterChanged, allowFreeDelivery_default,
    #     #              deliveryProportion_default,taxiProportion_default, UAVPerBase_default,BaseNum,
    #     #              taxiLeisureProbabilityMin_default,UAVEndurance_default,SecSpeed_default,detourTimeLimit_default,PTL_default,day))
    # print('base number costs', datetime.datetime.now() - functionBegins)
    #
    # for SecSpeed in [12,14,16,18,20]:
    #     parameterChanged = 'UAVSpeed'
    #     # 清空文件后再运行对应代码
    #     # delResultsFiles(parameterChanged,day)
    #     os.system('python 2-cooperation.py %s %i %f %f %i %i %f %i %i %f %i %i' % (parameterChanged, allowFreeDelivery_default,
    #                  deliveryProportion_default,taxiProportion_default, UAVPerBase_default,baseNum_default,
    #                  taxiLeisureProbabilityMin_default,UAVEndurance_default,SecSpeed,detourTimeLimit_default,PTL_default,day))
    #     os.system('python 2-cooperation.py %s %i %f %f %i %i %f %i %i %f %i %i' % (parameterChanged, 0,
    #                  deliveryProportion_default,taxiProportion_default, UAVPerBase_default,baseNum_default,
    #                  taxiLeisureProbabilityMin_default,UAVEndurance_default,SecSpeed,detourTimeLimit_default,PTL_default,day))
    #     # os.system('python 2-onlyUAV.py %s %i %f %f %i %i %f %i %i %f %i %i' % (parameterChanged, allowFreeDelivery_default,
    #     #              deliveryProportion_default,taxiProportion_default, UAVPerBase_default,baseNum_default,
    #     #              taxiLeisureProbabilityMin_default,UAVEndurance_default,SecSpeed,detourTimeLimit_default,PTL_default,day))
    # print('UAV speed costs', datetime.datetime.now() - functionBegins)
    #
    # for detourTime in [3,6,9,12]:
    #     parameterChanged = 'detourTime'
    #     # 清空文件后再运行对应代码
    #     # delResultsFiles(parameterChanged,day)
    #     os.system('python 2-cooperation.py %s %i %f %f %i %i %f %i %i %f %i %i' % (parameterChanged, allowFreeDelivery_default,
    #                  deliveryProportion_default,taxiProportion_default, UAVPerBase_default,baseNum_default,
    #                  taxiLeisureProbabilityMin_default,UAVEndurance_default,SecSpeed_default,detourTime,PTL_default, day))
    #     os.system('python 2-cooperation.py %s %i %f %f %i %i %f %i %i %f %i %i' % (parameterChanged, 0,
    #                  deliveryProportion_default,taxiProportion_default, UAVPerBase_default,baseNum_default,
    #                  taxiLeisureProbabilityMin_default,UAVEndurance_default,SecSpeed_default,detourTime,PTL_default, day))
    #     os.system('python 2-onlyTaxi.py %s %i %f %f %i %i %f %i %i %f %i %i' % (parameterChanged, allowFreeDelivery_default,
    #                  deliveryProportion_default,taxiProportion_default, UAVPerBase_default,baseNum_default,
    #                  taxiLeisureProbabilityMin_default,UAVEndurance_default,SecSpeed_default,detourTime,PTL_default, day))
    #     os.system('python 2-onlyTaxi.py %s %i %f %f %i %i %f %i %i %f %i %i' % (parameterChanged, 0,
    #                  deliveryProportion_default,taxiProportion_default, UAVPerBase_default,baseNum_default,
    #                  taxiLeisureProbabilityMin_default,UAVEndurance_default,SecSpeed_default,detourTime,PTL_default, day))
    # print('detour time costs', datetime.datetime.now() - functionBegins)

    for PTL in [3,4,5,6]:
        parameterChanged = 'PTL'
        # 清空文件后再运行对应代码
        # delResultsFiles(parameterChanged,day)
        os.system('python 2-cooperation.py %s %i %f %f %i %i %f %i %i %f %i %i' % (parameterChanged, allowFreeDelivery_default,
                     deliveryProportion_default,taxiProportion_default, UAVPerBase_default,baseNum_default,
                     taxiLeisureProbabilityMin_default,UAVEndurance_default,SecSpeed_default,detourTimeLimit_default,PTL, day))
        os.system('python 2-cooperation.py %s %i %f %f %i %i %f %i %i %f %i %i' % (parameterChanged, 0,
                     deliveryProportion_default,taxiProportion_default, UAVPerBase_default,baseNum_default,
                     taxiLeisureProbabilityMin_default,UAVEndurance_default,SecSpeed_default,detourTimeLimit_default,PTL, day))
        os.system('python 2-onlyTaxi.py %s %i %f %f %i %i %f %i %i %f %i %i' % (parameterChanged, allowFreeDelivery_default,
                     deliveryProportion_default,taxiProportion_default, UAVPerBase_default,baseNum_default,
                     taxiLeisureProbabilityMin_default,UAVEndurance_default,SecSpeed_default,detourTimeLimit_default,PTL, day))
        os.system('python 2-onlyTaxi.py %s %i %f %f %i %i %f %i %i %f %i %i' % (parameterChanged, 0,
                     deliveryProportion_default,taxiProportion_default, UAVPerBase_default,baseNum_default,
                     taxiLeisureProbabilityMin_default,UAVEndurance_default,SecSpeed_default,detourTimeLimit_default,PTL, day))
    print('PTL costs', datetime.datetime.now() - functionBegins)



if __name__ == '__main__':
    startTime = datetime.datetime.now()
    for day in range(2,3):
        startDayAt = datetime.datetime.now()
        print('day', day, 'start at',startDayAt)
        calculateResults(day)
    print('calculation costs', datetime.datetime.now() - startTime)
