import pandas as pd
import numpy as np
import random
import re,os
from pycaret.classification import *
from PycaretWrapper import PycaretWrapper
from IQUPpredict import Predict
from IQUPscoring import Scoring
# from MLProcess.Stacking import Stacking
# from MLProcess.Voting import Voting
from IQUPdrawPlot import DrawPlot
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from collections import Counter
import psutil
from sklearn.metrics import classification_report
import time


def changeBinaryFeatureInDf(dataDf):
    """
    因為 LightGBM 無法接受 binary feature 以及 int64 type 的 feature, 所以人工處理.
    binary feature: 隨機挑 1% 來改變數值 (加減 0.01 or 乘上 0.99/1.01)
    int64 type feature: 手動轉換為 float64 type
    :param dataDf:
    :return:
    """
    pd.options.mode.chained_assignment = None
    threshold = len(dataDf.index.tolist()) * (95 / 100)
    for column in dataDf.columns.to_list():
        dfUniqueValue = dataDf[column].unique()
        count_class = Counter(dataDf[column])
        dfCounterValue = pd.DataFrame.from_dict(count_class, orient='index', columns=["Count"])
        dfCounterValueSum = dfCounterValue['Count'].nlargest(2).sum()
        if (dfCounterValueSum >= threshold) and (column != 'y'):
            value1 = dfCounterValue['Count'].nlargest(2).index.tolist()[0]
            value2 = dfCounterValue['Count'].nlargest(2).index.tolist()[1]
            if 0 in dfUniqueValue:
                convertMaxValue = value1 - 0.01
                convertMinValue = value2 + 0.01
                print('Value of ' + str(column) + ' Converted')
                print(str(value1) + '  --+0.01-->  ' + str(convertMaxValue))
                print(str(value2) + '  --+0.01-->  ' + str(convertMinValue))
            else:
                convertMaxValue = value1 * 0.99
                convertMinValue = value2 * 1.01
                print('Value of ' + str(column) + ' Converted')
                print(str(value1) + '  --*0.99-->  ' + str(convertMaxValue))
                print(str(value2) + '  --*1.01-->  ' + str(convertMinValue))
            value1Index = list(dataDf.loc[dataDf[column] == value1].index[:])
            value2Index = list(dataDf.loc[dataDf[column] == value2].index[:])
            countMax_int = int(np.ceil(len(value1Index) * 0.1))
            countMin_int = int(np.ceil(len(value2Index) * 0.1))
            randomMaxIndex = random.sample(value1Index, countMax_int)
            randomMinIndex = random.sample(value2Index, countMin_int)
            dataDf[column].loc[randomMaxIndex] = dataDf[column].loc[randomMaxIndex].replace(value1, convertMaxValue)
            dataDf[column].loc[randomMinIndex] = dataDf[column].loc[randomMinIndex].replace(value2, convertMinValue)
        if (dataDf[column].dtype.name == 'int64') and (column != 'y'):
            dataDf[column] = dataDf[column].astype('float64')
            print('Type of ' + str(column) + ' Converted')
            print('int64  ---->  float64')

    return dataDf

def changeDataframeDataType(testDataDf):
    for column in testDataDf.columns.to_list():
        if (testDataDf[column].dtype.name == 'float64') and (column != 'y'):
            testDataDf[column] = testDataDf[column].astype('float16')
            print('Type of ' + str(column) + ' Converted')
            print('float64  ---->  float16')
        if column == 'y':
            testDataDf[column] = testDataDf[column].astype('int8')
    return testDataDf

# dataDict = {"Basel":"gbc","Lina":"lightgbm","NCI":"catboost"}
# data,model = list(dataDict.items())[0]

dataList = ["Basel","Lina","NCI"]
catFeatures = ['Mass', 'PTMCnt', 'EucliIntraProteinPair', 'Charge2', 'Charge3', 'Charge4', 'Charge5', 'PTMCntRatio',
               'PeptideLength', 'CosineIntraProtein', 'MassDifferent', 'AvgIntensityLog', 'expect', 'fval',
               'EucliIntraProtein', 'ManhattanIntraProtein', 'PearsonIntraProtein', 'ManhattanIntraProteinPair',
               'CosineIntraProteinPair', 'y']

testFeatures = ["Name",'Mass', 'PTMCnt', 'EucliIntraProteinPair', 'Charge2', 'Charge3', 'Charge4', 'Charge5', 'PTMCntRatio',
               'PeptideLength', 'CosineIntraProtein', 'MassDifferent', 'AvgIntensityLog', 'expect', 'fval',
               'EucliIntraProtein', 'ManhattanIntraProtein', 'PearsonIntraProtein', 'ManhattanIntraProteinPair',
               'CosineIntraProteinPair', 'y']

modelNameList = ['lightgbm', 'catboost', 'gbc', 'nb', 'et', 'rf', 'qda',
                 'xgboost']  # 'ridge','lr', 'lda','ada', 'knn', 'mlp',  'dt',svm gpc rbfsvm拿掉

pycObj = PycaretWrapper()

for data in dataList:
    Datapath = f"D:/IQUPexperiment/MlData/先做Normalization再做Smote/{data}/"
    tuneModelPath = "D:/IQUPexperiment/data/tuneModel"
    scorePath = f"D:/IQUPexperiment/data/0614分析/{data}/"
    finalModelPath = "D:/IQUPexperiment/data/finalModel"
    paramPath = "D:/IQUPexperiment/data/param/"
    w_path = f"D:/IQUPexperiment/data/0614分析/{data}/"

    csvTestFilenames = [f'StandarTest{data}.csv']

# foundCharge = False

#Train
# for trainFilename in csvTrainFilenames:
    # for i in range(len(catFeatures)):
    #     if catFeatures[i] == "y":
    #         break
    #     if foundCharge == True:
    #         break
    #     if catFeatures[i] != 'y':
    #         trainFeatureList = catFeatures.copy()
    #         delFeatureName = catFeatures[i]
    #         del trainFeatureList[i]
    #     if "Charge" in catFeatures[i] and not foundCharge:
    #         delFeatureName = "Charge"
    #         trainFeatureList = [feature for feature in catFeatures if "Charge" not in feature]
    #         foundCharge = True

    trainDataDf = pd.read_csv(f"{Datapath}StandarTrain{data}.csv", usecols=catFeatures)

    train_X = trainDataDf.drop(['y'], axis=1)
    train_y = trainDataDf[["y"]]

    trainDataDf = trainDataDf.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    trainDataDf = changeBinaryFeatureInDf(trainDataDf)
    setupDf = pycObj.doSetup(trainData=trainDataDf)

    tunedModelList, tunerList = pycObj.doTuneModel(searchLibrary='optuna', searchAlg='tpe', includeModelList=modelNameList, foldNum=5,
                       n_iter=10, early_stopping=False, customGridDict=None,) #n_iter用50

    pycObj.doSaveModel(tuneModelPath, b_isFinalizedModel=False)
    loadtuneModelList = pycObj.doLoadModel(tuneModelPath, b_isFinalizedModel=False)
    tunedModelParamList, scoreRank = pycObj.doCompareModel(fold=5,
                                                               includeModelList=loadtuneModelList)
    scoreRank.to_csv(f"{scorePath}StandTrain{data}Score.csv")

    print("Train is Done!")

    for testFilename in csvTestFilenames:
        testDataDf = pd.read_csv(f"{Datapath}{testFilename}",usecols=testFeatures)
        testNameDf = testDataDf[['Name']]
        testDataDf = testDataDf.drop(['Name'], axis=1)
        testDataDf = testDataDf.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
        testDataDf = changeBinaryFeatureInDf(testDataDf)

        if 'NCI' in testFilename:
            testDataDf = changeDataframeDataType(testDataDf)
        else:
            testDataDf = testDataDf

        testDataDf_X = testDataDf.drop(['y'], axis=1)
        testDataDf_y = testDataDf[["y"]]
        pycObj.doFinalizeModel(tunedModelList)
        pycObj.doSaveModel(finalModelPath, b_isFinalizedModel=True)
        finalModelList = pycObj.doLoadModel(path=finalModelPath,fileNameList=modelNameList, b_isFinalizedModel=True)
        predObjIndp = Predict(dataX=testDataDf_X, modelList=finalModelList)
        predVectorListIndp, probVectorListIndp = predObjIndp.doPredict()
        probVectorDf = pd.DataFrame(probVectorListIndp, index=modelNameList, columns=testDataDf_y.index).T
        probVectorDf.to_csv(scorePath + f'probVectorStandarTrain{data}{testFilename.split(".csv")[0]}.csv')
        scoreObjIndp = Scoring(predVectorList=predVectorListIndp, probVectorList=probVectorListIndp,
                               answerDf=testDataDf_y, modelNameList=modelNameList)
        bestProArrDict,bestPredArrDict,bestPredVectorListIndp,bestpredVactor,bestprobVactor = scoreObjIndp.optimizeMcc(cutOffList=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                                                          # bestPredVectorListIndp：probVectorListIndp 經過最佳 cutoff 轉出的 0 or 1 vector (binary prediction)
                                                          method='mcc',  # 加在技術文件說明
                                                          bestCutoffJsonPath=f'{paramPath}/bestCutoff.json')
        scoreDfIndp = scoreObjIndp.doScoring(b_optimizedMcc=True, path=scorePath + f'singleModelScore_IndpStandarTrain{data}{testFilename.split(".csv")[0]}.csv',
                                             sortColumn='mcc')
        for model in modelNameList:
            preDataDf = pd.concat([testNameDf,probVectorDf[model]], axis= 1)
            preDataDf[f"{model}predict"] = bestPredArrDict[model]
            preDataDf.to_csv(f"{w_path}{model}predictAnswer.csv")
        print("Test is Done !")
        # preDataDf = pd.concat([testNameDf,testDataDf_X, testDataDf_y], axis=1)
        # preDataDf['predict'] = bestpredVactor
        # preDataDf['probablity of 1'] = bestprobVactor
        # zeroProVactor = []
        # for prob in bestprobVactor:
        #     zeroProb = float(1-float(prob))
        #     zeroProVactor.append(zeroProb)
        # preDataDf['probablity of 0'] = zeroProVactor
        # preDataDf.to_csv(f'{w_path}predictAnswer')
    #     scoreObjIndp.plotPredConfidence(predictionsList=probVectorListIndp, trueLabelsDf=testDataDf_y, numBins=10,
    #                                     modelNameList=modelNameList,
    #                                     outputExcel=scorePath + f'plotPredConfidence{trainFilename.split(".csv")[0]}{testFilename.split(".csv")[0]}.xlsx',
    #                                     figSave=True,
    #                                     figSavePath=scorePath)
    #     drawObj = DrawPlot(answerDf=testDataDf_y, modelList=finalModelList, modelNameList=modelNameList,
    #                        predArrList=predVectorListIndp, probArrList=probVectorListIndp)
    #     aucDf = drawObj.drawROC(colorList=None, title=False, titleName=f'Receiver Operating Characteristic{trainFilename.split(".csv")[0]}{testFilename.split(".csv")[0]}', setDpi=True,
    #                             legendSize=11, labelSize=20, save=True, saveLoc=scorePath + f'Multi_Single_Model{trainFilename.split(".csv")[0]}{testFilename.split(".csv")[0]}.png',
    #                             show=True,
    #                             dpi=300, figSize=(12, 9))

