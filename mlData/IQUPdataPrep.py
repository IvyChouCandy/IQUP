import csv,os,re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN,BorderlineSMOTE,SVMSMOTE,SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.preprocessing import StandardScaler,MinMaxScaler

class prepData:
    def __init__(self):
        self.columnList = ['Name']

    def readtxt(self,filename):
        file = open(filename)
        for line in file.readlines():
            line = line.strip()
            line = line.split('  ')
            self.columnList.append(line[0])
        file.close()

    def readData(self,path,filename):
        dataDf = pd.read_csv(f"{path}{filename}")
        yDf = self.appendYvalue(dataDf)
        collectDataDf = dataDf[self.columnList]
        data_dum = pd.get_dummies(collectDataDf['Charge'])
        chargeDf = pd.DataFrame(data_dum)
        collectDataDf = collectDataDf.drop(columns=['Charge'])
        collectDataDf = pd.DataFrame(collectDataDf).join(chargeDf)
        collectDataDf = pd.merge(collectDataDf,yDf,on = 'Name')
        collectDataDf = collectDataDf.rename(columns={2:"Charge2",3:'Charge3',4:'Charge4',5:'Charge5'})
        return collectDataDf

    def appendYvalue(self,dataDf):
        yValue = {'y':[],'Name':[]}
        ratioCountDf = dataDf[['Name','RatioCount','UnreliableCnt']]
        for index,row in ratioCountDf.iterrows():
            ratioCount = int(row['RatioCount'])
            unreliable = int(row['UnreliableCnt'])
            name = row['Name']
            y = float(unreliable/ratioCount)
            if y >0.5:
                yValue['y'].append('1')
                yValue['Name'].append(name)
            if y<= 0.5:
                yValue['y'].append('0')
                yValue['Name'].append(name)
        yDf = pd.DataFrame(yValue)
        return yDf

    def standarData(self,trainDf,testDf,wPath,wFilename):
        trainDfIndex = trainDf.index.to_list()
        trainDfAnswer = trainDf[["Name",'Charge2','Charge3','Charge4','Charge5','y']]
        trainDfFeature = trainDf.drop(['Name','Charge2','Charge3','Charge4','Charge5','y'],axis=1)
        trainDfFeatureCol = trainDfFeature.columns.to_list()
        trainArray = trainDfFeature.values
        # ==============================================================
        testDfIndex = testDf.index.to_list()
        testDfAnswer = testDf[['Name','Charge2','Charge3','Charge4','Charge5','y']]
        testDfFeature = testDf.drop(['Name','Charge2','Charge3','Charge4','Charge5','y'], axis=1)
        testDfFeatureCol = testDfFeature.columns.to_list()
        testArray = testDfFeature.values
        #存成scaler並測試
        self.scaler = StandardScaler()
        scalerTrainDf = self.scaler.fit_transform(trainArray)
        scalerTrainDf = pd.DataFrame(scalerTrainDf,index=trainDfIndex,columns=trainDfFeatureCol)
        scalerTrainDf = scalerTrainDf.join(trainDfAnswer)
        scalerTestDf = self.scaler.transform(testArray)
        scalerTestDf = pd.DataFrame(scalerTestDf,index=testDfIndex,columns=testDfFeatureCol)
        scalerTestDf = scalerTestDf.join(testDfAnswer)
        scalerTrainDf.to_csv(f"{wPath}StandarTrain{wFilename}")
        scalerTestDf.to_csv(f"{wPath}StandarTest{wFilename}")
        return scalerTrainDf,scalerTestDf

    def smoteNormalization(self,smoteDf):
        smoteDfIndex = smoteDf.index.to_list()
        smoteDfAnswer = smoteDf[['Charge2','Charge3','Charge4','Charge5','y']]
        smoteDfFeature = smoteDf.drop(['Charge2', 'Charge3', 'Charge4', 'Charge5','y'], axis=1)
        smoteDfFeatureCol = smoteDfFeature.columns.to_list()
        smoteDfArray = smoteDfFeature.values
        scalersmoteDf = self.scaler.transform(smoteDfArray)
        scalersmoteDf = pd.DataFrame(scalersmoteDf, index=smoteDfIndex, columns=smoteDfFeatureCol)
        scalersmoteDf = scalersmoteDf.join(smoteDfAnswer)
        return scalersmoteDf

    def dataDfNormalization(self,dataDf,wPath,wFilename):
        dataDfIndex = dataDf.index.to_list()
        dataDfAnswer = dataDf[['Charge2','Charge3','Charge4','Charge5','y']]
        dataDfFeature = dataDf.drop(['Charge2', 'Charge3', 'Charge4', 'Charge5','y'], axis=1)
        dataDfFeatureCol = dataDfFeature.columns.to_list()
        dataDfArray = dataDfFeature.values
        scaler = StandardScaler()
        scalerDataDf = scaler.fit_transform(dataDfArray)
        scalerDataDf = pd.DataFrame(scalerDataDf,index=dataDfIndex,columns=dataDfFeatureCol)
        scalerDataDf = scalerDataDf.join(dataDfAnswer)
        scalerDataDf.to_csv(f"{wPath}{wFilename}")
        return scalerDataDf

    def smoteMinMax(self,smoteDf):
        smoteDfIndex = smoteDf.index.to_list()
        smoteDfAnswer = smoteDf[['Charge2','Charge3','Charge4','Charge5','y']]
        smoteDfFeature = smoteDf.drop(['Charge2', 'Charge3', 'Charge4', 'Charge5','y'], axis=1)
        smoteDfFeatureCol = smoteDfFeature.columns.to_list()
        smoteDfArray = smoteDfFeature.values
        scalersmoteDf = self.minMaxSca.transform(smoteDfArray)
        scalersmoteDf = pd.DataFrame(scalersmoteDf, index=smoteDfIndex, columns=smoteDfFeatureCol)
        scalersmoteDf = scalersmoteDf.join(smoteDfAnswer)
        return scalersmoteDf

    def noNormalization(self,oriTrain,dataRatioDict,csvFilename,oriSmoteDataPath):
        smoteModelList = ['ADASYN', 'BorderlineSMOTE', 'SVMSMOTE', 'SMOTE']
        dataRatioList = dataRatioDict[csvFilename]
        X_train = oriTrain.drop(columns=['y'])
        y_train = oriTrain['y']
        for sampleRatio in dataRatioList:
            for smoteModel in smoteModelList:
                w_Filename = f'{oriSmoteDataPath}{round(sampleRatio, 3)}Train{smoteModel}{csvFilename.split("Classifyout")[1]}'
                X_smote_train, y_smote_train = smoteData.smoteAllModelData(X_train, y_train, smoteModel, sampleRatio)
                # 計算postive&negative
                # postive, negative = smoteDataObj.checkDataRatio(y_smote_train)
                # dataCountDict[w_Filename] = [str(postive), str(negative)]
                smoteDf = pd.concat([X_smote_train, y_smote_train], axis=1)
                smoteDf.to_csv(w_Filename)
        # 將positve&negative的筆數寫成檔案
        # with open('TrainValuesCount.csv', 'w', newline='') as csvfile:
        #     csvwriter = csv.writer(csvfile)
        #     csvwriter.writerow(["Filename", "PositiveValue", "NegativeValue"])
        #     for key, value in dataCountDict.items():
        #         csvwriter.writerow([key] + value)

    def doNormalization(self,oriSmoteDataPath,oriSmotefilename,w_StandarDataPath):
        #讀不做normalization後經過smote的棒案作normalize
        oriDataDf = pd.read_csv(f"{oriSmoteDataPath}{oriSmotefilename}",usecols=['Mass','PTMCnt','PTMCntRatio','MassDifferent','PeptideLength','AvgIntensityLog','expect','fval','EucliIntraProtein','ManhattanIntraProtein','PearsonIntraProtein','CosineIntraProtein','EucliIntraProteinPair','ManhattanIntraProteinPair','CosineIntraProteinPair','Charge2','Charge3','Charge4','Charge5','y'])
        scalersmoteDf = self.smoteNormalization(oriDataDf)
        scalersmoteDf.to_csv(f"{w_StandarDataPath}Standar{oriSmotefilename}")

    def doMinMax(self,oriSmoteDataPath,oriSmotefilename,w_MinMaxDataPath):
        oriDataDf = pd.read_csv(f"{oriSmoteDataPath}{oriSmotefilename}",usecols=['Mass','PTMCnt','PTMCntRatio','MassDifferent','PeptideLength','AvgIntensityLog','expect','fval','EucliIntraProtein','ManhattanIntraProtein','PearsonIntraProtein','CosineIntraProtein','EucliIntraProteinPair','ManhattanIntraProteinPair','CosineIntraProteinPair','Charge2','Charge3','Charge4','Charge5','y'])
        scalersmoteDf = self.smoteMinMax(oriDataDf)
        scalersmoteDf.to_csv(f"{w_MinMaxDataPath}MinMax{oriSmotefilename}")

    def minMaxData(self,trainDf,testDf):
        trainDfIndex = trainDf.index.to_list()
        trainDfAnswer = trainDf[['Charge2', 'Charge3', 'Charge4', 'Charge5', 'y']]
        trainDfFeature = trainDf.drop(['Charge2', 'Charge3', 'Charge4', 'Charge5', 'y'], axis=1)
        trainDfFeatureCol = trainDfFeature.columns.to_list()
        trainArray = trainDfFeature.values
        # ==============================================================
        testDfIndex = testDf.index.to_list()
        testDfAnswer = testDf[['Charge2', 'Charge3', 'Charge4', 'Charge5', 'y']]
        testDfFeature = testDf.drop(['Charge2', 'Charge3', 'Charge4', 'Charge5', 'y'], axis=1)
        testDfFeatureCol = testDfFeature.columns.to_list()
        testArray = testDfFeature.values

        self.minMaxSca = MinMaxScaler()
        scalerTrainDf = self.minMaxSca.fit_transform(trainArray)
        scalerTrainDf = pd.DataFrame(scalerTrainDf, index=trainDfIndex, columns= trainDfFeatureCol)
        scalerTrainDf = scalerTrainDf.join(trainDfAnswer)
        scalerTestDf = self.minMaxSca.fit_transform(testArray)
        scalerTestDf = pd.DataFrame(scalerTestDf, index=testDfIndex, columns=testDfFeatureCol)
        scalerTestDf = scalerTestDf.join(testDfAnswer)
        return scalerTrainDf, scalerTestDf

class smoteData:
    def checkDataRatio(self,checkDataDf):
        plt.figure(figsize=(10,5))
        checkDataDf.value_counts('y').plot(kind = 'pie',colors=['lightcoral','skyblue'],autopct='%1.2f%%')
        countSer = checkDataDf.value_counts('y')
        postive = countSer[1]
        negative = countSer[0]
        print(f'postive{postive}')
        print(f'negative{negative}')
        plt.title('Postive/Negative')
        plt.ylabel(' ')
        plt.show()


    def writeReData(self,XDf,yDf,w_Filename):
        reDataDf = pd.concat([XDf,yDf], axis=1)
        reDataDf.to_csv(w_Filename)

    def smoteAllModelData(self,X_train,y_train,smoteModel,sampleRatio):
        if smoteModel == 'ADASYN':
            adasynobj = ADASYN(sampling_strategy=sampleRatio)
            X_smote_train, y_smote_train = adasynobj.fit_resample(X_train,y_train)
        if smoteModel == 'BorderlineSMOTE':
            BorderlineSMOTEobj = BorderlineSMOTE(sampling_strategy=sampleRatio)
            X_smote_train, y_smote_train = BorderlineSMOTEobj.fit_resample(X_train, y_train)
        if smoteModel == 'SVMSMOTE':
            SVMSMOTEObj = SVMSMOTE(sampling_strategy=sampleRatio)
            X_smote_train, y_smote_train = SVMSMOTEObj.fit_resample(X_train, y_train)
        if smoteModel == 'SMOTE':
            SMOTEObj = SMOTE(sampling_strategy=sampleRatio)
            X_smote_train, y_smote_train = SMOTEObj.fit_resample(X_train, y_train)
        return X_smote_train,y_smote_train


# dataPath = "D:/IQUPexperiment/originalDataSet/"
# w_dataPath = "D:/IQUPexperiment/MlData/先做Normalization再做Smote/NCI"
smoteObj = smoteData()
prepDataObj = prepData()
#
# filename = 'ClassifyoutNCI.csv'
# prepDataObj.readtxt('Column to analyze.txt')
# collectDataDf = prepDataObj.readData(dataPath,filename)
# collectDataDf.to_csv(f'{dataPath}ori{filename.split("Classifyout")[1]}')
#
#
# if not os.path.isdir(w_dataPath):
#     os.makedirs(w_dataPath)
#
# #先做normalization
# oriDataDf = pd.read_csv(f"{dataPath}oriNCI.csv",usecols=['Name','Mass','PTMCnt','PTMCntRatio','MassDifferent','PeptideLength','AvgIntensityLog','expect','fval','EucliIntraProtein','ManhattanIntraProtein','PearsonIntraProtein','CosineIntraProtein','EucliIntraProteinPair','ManhattanIntraProteinPair','CosineIntraProteinPair','Charge2','Charge3','Charge4','Charge5','y'])
#
OritrainDf, OritestDf = train_test_split(oriDataDf, test_size=0.2, shuffle=True)
#
# OritrainDf.to_csv(f'{w_dataPath}/OriTrain{filename.split("Classifyout")[1]}')
# OritestDf.to_csv(f'{w_dataPath}/OriTest{filename.split("Classifyout")[1]}')
#
# trainDfIndex = OritrainDf.index.to_list()
# trainDfAnswer = OritrainDf[['Name','Charge2', 'Charge3', 'Charge4', 'Charge5', 'y']]
# trainDfFeature = OritrainDf.drop(['Name','Charge2', 'Charge3', 'Charge4', 'Charge5', 'y'], axis=1)
# trainDfFeatureCol = trainDfFeature.columns.to_list()
# trainArray = trainDfFeature.values
# # ==============================================================
# testDfIndex = OritestDf.index.to_list()
# testDfAnswer = OritestDf[['Name','Charge2', 'Charge3', 'Charge4', 'Charge5', 'y']]
# testDfFeature = OritestDf.drop(['Name','Charge2', 'Charge3', 'Charge4', 'Charge5', 'y'], axis=1)
# testDfFeatureCol = testDfFeature.columns.to_list()
# testArray = testDfFeature.values
# # 存成scaler並測試
# scaler = StandardScaler()
# scalerTrainDf = scaler.fit_transform(trainArray)
# scalerTrainDf = pd.DataFrame(scalerTrainDf, index=trainDfIndex, columns=trainDfFeatureCol)
# scalerTrainDf = scalerTrainDf.join(trainDfAnswer)
# scalerTestDf = scaler.transform(testArray)
# scalerTestDf = pd.DataFrame(scalerTestDf, index=testDfIndex, columns=testDfFeatureCol)
# scalerTestDf = scalerTestDf.join(testDfAnswer)
#
# scalerTrainDf.to_csv(f'{w_dataPath}/StandarTrain{filename.split("Classifyout")[1]}')
# scalerTestDf.to_csv(f'{w_dataPath}/StandarTest{filename.split("Classifyout")[1]}')




#mergeData
readPath = "D:/IQUPexperiment/MlData/先做Normalization再做Smote/"
wpath = "D:/IQUPexperiment/MlData/Merge/"
baselTrainDf = pd.read_csv(f"{readPath}Basel/OriTrainBasel.csv",usecols=['Name','Mass','PTMCnt','PTMCntRatio','MassDifferent','PeptideLength','AvgIntensityLog','expect','fval','EucliIntraProtein','ManhattanIntraProtein','PearsonIntraProtein','CosineIntraProtein','EucliIntraProteinPair','ManhattanIntraProteinPair','CosineIntraProteinPair','Charge2','Charge3','Charge4','Charge5','y'])
linaTrainDf = pd.read_csv(f"{readPath}Lina/OriTrainLina.csv",usecols=['Name','Mass','PTMCnt','PTMCntRatio','MassDifferent','PeptideLength','AvgIntensityLog','expect','fval','EucliIntraProtein','ManhattanIntraProtein','PearsonIntraProtein','CosineIntraProtein','EucliIntraProteinPair','ManhattanIntraProteinPair','CosineIntraProteinPair','Charge2','Charge3','Charge4','Charge5','y'])
nciTrainDf = pd.read_csv(f"{readPath}NCI/OriTrainNCI.csv",usecols=['Name','Mass','PTMCnt','PTMCntRatio','MassDifferent','PeptideLength','AvgIntensityLog','expect','fval','EucliIntraProtein','ManhattanIntraProtein','PearsonIntraProtein','CosineIntraProtein','EucliIntraProteinPair','ManhattanIntraProteinPair','CosineIntraProteinPair','Charge2','Charge3','Charge4','Charge5','y'])

baselfilt = (baselTrainDf["y"] == 1)
baselPostiveNumberDf = baselTrainDf[baselfilt]
MergePostiveNum = len(baselPostiveNumberDf["y"])

linafilt = (linaTrainDf["y"] == 0)
linaNegativeNumberDf = linaTrainDf[linafilt]
MergeNegativeNum = len(linaNegativeNumberDf["y"])

BPfilt = (baselTrainDf["y"] == 1)
BNfilt = (baselTrainDf["y"] == 0)
LPfilt = (linaTrainDf["y"] == 1)
LNfilt = (linaTrainDf["y"] == 0)
NPfilt = (nciTrainDf["y"] == 1)
NNfilt = (nciTrainDf["y"] == 0)
mergeBPTDf = baselTrainDf[BPfilt].sample(n=MergePostiveNum)
mergeBNTDf = baselTrainDf[BNfilt].sample(n=MergeNegativeNum)
mergeLPTDf = linaTrainDf[LPfilt].sample(n=MergePostiveNum)
mergeLNTDf = linaTrainDf[LNfilt].sample(n=MergeNegativeNum)
mergeNPTDf = nciTrainDf[NPfilt].sample(n=MergePostiveNum)
mergeNNTDf = nciTrainDf[NNfilt].sample(n=MergeNegativeNum)

dfs = [df for df in [mergeBPTDf,mergeBNTDf,mergeLPTDf,mergeLNTDf,mergeNPTDf,mergeNNTDf]]
mergeDf = pd.concat(dfs,axis=0)
mergeDf.to_csv(f"{wpath}OriTrainMerge.csv")

baselTestDf = pd.read_csv(f"{readPath}Basel/OriTestBasel.csv",usecols=['Name','Mass','PTMCnt','PTMCntRatio','MassDifferent','PeptideLength','AvgIntensityLog','expect','fval','EucliIntraProtein','ManhattanIntraProtein','PearsonIntraProtein','CosineIntraProtein','EucliIntraProteinPair','ManhattanIntraProteinPair','CosineIntraProteinPair','Charge2','Charge3','Charge4','Charge5','y'])
linaTestDf = pd.read_csv(f"{readPath}Lina/OriTestLina.csv",usecols=['Name','Mass','PTMCnt','PTMCntRatio','MassDifferent','PeptideLength','AvgIntensityLog','expect','fval','EucliIntraProtein','ManhattanIntraProtein','PearsonIntraProtein','CosineIntraProtein','EucliIntraProteinPair','ManhattanIntraProteinPair','CosineIntraProteinPair','Charge2','Charge3','Charge4','Charge5','y'])
nciTestDf = pd.read_csv(f"{readPath}NCI/OriTestNCI.csv",usecols=['Name','Mass','PTMCnt','PTMCntRatio','MassDifferent','PeptideLength','AvgIntensityLog','expect','fval','EucliIntraProtein','ManhattanIntraProtein','PearsonIntraProtein','CosineIntraProtein','EucliIntraProteinPair','ManhattanIntraProteinPair','CosineIntraProteinPair','Charge2','Charge3','Charge4','Charge5','y'])

mergeDfIndex = mergeDf.index.to_list()
mergeDfAnswer = mergeDf[["Name", 'Charge2', 'Charge3', 'Charge4', 'Charge5', 'y']]
mergeDfFeature = mergeDf.drop(['Name', 'Charge2', 'Charge3', 'Charge4', 'Charge5', 'y'], axis=1)
mergeDfFeatureCol = mergeDfFeature.columns.to_list()
mergeDftrainArray = mergeDfFeature.values
# ==============================================================
baselTestDfIndex = baselTestDf.index.to_list()
baselTestDfAnswer = baselTestDf[['Name', 'Charge2', 'Charge3', 'Charge4', 'Charge5', 'y']]
baselTestDfFeature = baselTestDf.drop(['Name', 'Charge2', 'Charge3', 'Charge4', 'Charge5', 'y'], axis=1)
baselTestDfFeatureCol = baselTestDfFeature.columns.to_list()
baselTestDftestArray = baselTestDfFeature.values
# ==============================================================
linaTestDfIndex = linaTestDf.index.to_list()
linaTestDfAnswer = linaTestDf[['Name', 'Charge2', 'Charge3', 'Charge4', 'Charge5', 'y']]
linaTestDfFeature = linaTestDf.drop(['Name', 'Charge2', 'Charge3', 'Charge4', 'Charge5', 'y'], axis=1)
linaTestDfFeatureCol = linaTestDfFeature.columns.to_list()
linaTestDftestArray = linaTestDfFeature.values
# ==============================================================
nciTestDfIndex = nciTestDf.index.to_list()
nciTestDfAnswer = nciTestDf[['Name', 'Charge2', 'Charge3', 'Charge4', 'Charge5', 'y']]
nciTestDfFeature = nciTestDf.drop(['Name', 'Charge2', 'Charge3', 'Charge4', 'Charge5', 'y'], axis=1)
nciTestDfFeatureCol = nciTestDfFeature.columns.to_list()
nciTestDftestArray = nciTestDfFeature.values

# 存成scaler並測試
scaler = StandardScaler()
scalerTrainDf = scaler.fit_transform(mergeDftrainArray)
scalerTrainDf = pd.DataFrame(scalerTrainDf, index=mergeDfIndex, columns=mergeDfFeatureCol)
scalerTrainDf = scalerTrainDf.join(mergeDfAnswer)

scalerbaselTestDf = scaler.transform(baselTestDftestArray)
scalerbaselTestDf = pd.DataFrame(scalerbaselTestDf, index=baselTestDfIndex, columns=baselTestDfFeatureCol)
scalerbaselTestDf = scalerbaselTestDf.join(baselTestDfAnswer)

scalerlinaTestDf = scaler.transform(linaTestDftestArray)
scalerlinaTestDf = pd.DataFrame(scalerlinaTestDf, index=linaTestDfIndex, columns=linaTestDfFeatureCol)
scalerlinaTestDf = scalerlinaTestDf.join(linaTestDfAnswer)

scalernciTestDf = scaler.transform(nciTestDftestArray)
scalernciTestDf = pd.DataFrame(scalernciTestDf, index=nciTestDfIndex, columns=nciTestDfFeatureCol)
scalernciTestDf = scalernciTestDf.join(nciTestDfAnswer)

scalerTrainDf.to_csv(f"{wpath}StandarTrainMerge.csv")
scalerbaselTestDf.to_csv(f"{wpath}StandarTestBasel.csv")
scalerlinaTestDf.to_csv(f"{wpath}StandarTestLina.csv")
scalernciTestDf.to_csv(f"{wpath}StandarTestNCI.csv")

#使用TrainBasel對NCI、Lina做normalization
# readFileDataDf = pd.read_csv("D:/IQUPexperiment/MlData/先做Normalization再做Smote/NCI/OriTrainNCI.csv",usecols=['Name','Mass','PTMCnt','PTMCntRatio','MassDifferent','PeptideLength','AvgIntensityLog','expect','fval','EucliIntraProtein','ManhattanIntraProtein','PearsonIntraProtein','CosineIntraProtein','EucliIntraProteinPair','ManhattanIntraProteinPair','CosineIntraProteinPair','Charge2','Charge3','Charge4','Charge5','y'])
#
# testDf1 = pd.read_csv("D:/IQUPexperiment/originalDataSet/oriBasel.csv",usecols=['Name','Mass','PTMCnt','PTMCntRatio','MassDifferent','PeptideLength','AvgIntensityLog','expect','fval','EucliIntraProtein','ManhattanIntraProtein','PearsonIntraProtein','CosineIntraProtein','EucliIntraProteinPair','ManhattanIntraProteinPair','CosineIntraProteinPair','Charge2','Charge3','Charge4','Charge5','y'])
# testDf2 = pd.read_csv("D:/IQUPexperiment/originalDataSet/oriLina.csv",usecols=['Name','Mass','PTMCnt','PTMCntRatio','MassDifferent','PeptideLength','AvgIntensityLog','expect','fval','EucliIntraProtein','ManhattanIntraProtein','PearsonIntraProtein','CosineIntraProtein','EucliIntraProteinPair','ManhattanIntraProteinPair','CosineIntraProteinPair','Charge2','Charge3','Charge4','Charge5','y'])
#
# trainDfIndex = readFileDataDf.index.to_list()
# trainDfAnswer = readFileDataDf[['Name','Charge2', 'Charge3', 'Charge4', 'Charge5', 'y']]
# trainDfFeature = readFileDataDf.drop(['Name','Charge2', 'Charge3', 'Charge4', 'Charge5', 'y'], axis=1)
# trainDfFeatureCol = trainDfFeature.columns.to_list()
# trainArray = trainDfFeature.values
# # ==============================================================
# testDf1Index = testDf1.index.to_list()
# testDf1Answer = testDf1[['Name','Charge2', 'Charge3', 'Charge4', 'Charge5', 'y']]
# testDf1Feature = testDf1.drop(['Name','Charge2', 'Charge3', 'Charge4', 'Charge5', 'y'], axis=1)
# testDf1FeatureCol = testDf1Feature.columns.to_list()
# test1Array = testDf1Feature.values
# # ===============================================================
# testDf2Index = testDf2.index.to_list()
# testDf2Answer = testDf2[['Name','Charge2', 'Charge3', 'Charge4', 'Charge5', 'y']]
# testDf2Feature = testDf2.drop(['Name','Charge2', 'Charge3', 'Charge4', 'Charge5', 'y'], axis=1)
# testDf2FeatureCol = testDf2Feature.columns.to_list()
# test2Array = testDf2Feature.values
#
# # 存成scaler並測試
# scaler = StandardScaler()
# scalerTrainDf = scaler.fit_transform(trainArray)
# scalerTrainDf = pd.DataFrame(scalerTrainDf, index=trainDfIndex, columns=trainDfFeatureCol)
# scalerTrainDf = scalerTrainDf.join(trainDfAnswer)
# scalerTestDf1 = scaler.transform(test1Array)
# scalerTestDf1 = pd.DataFrame(scalerTestDf1, index=testDf1Index, columns=testDf1FeatureCol)
# scalerTestDf1 = scalerTestDf1.join(testDf1Answer)
# scalerTestDf2 = scaler.transform(test2Array)
# scalerTestDf2 = pd.DataFrame(scalerTestDf2, index=testDf2Index, columns=testDf2FeatureCol)
# scalerTestDf2 = scalerTestDf2.join(testDf2Answer)
#
# scalerTrainDf.to_csv(f'D:/IQUPexperiment/MlData/互相/StandarTrainNCI.csv')
# scalerTestDf1.to_csv(f'D:/IQUPexperiment/MlData/互相/StandarTestBasel.csv')
# scalerTestDf2.to_csv(f'D:/IQUPexperiment/MlData/互相/StandarTestLina.csv')

# #隨機取10%trainNCI
#
# nciDf = pd.read_csv(f'D:/IQUPexperiment/MlData/先做Normalization再做Smote/NCI/StandarTrainNCI.csv')
#
# for i in range(5):
#     subset = nciDf.sample(frac=0.1)
#     subset.to_csv(f"D:/IQUPexperiment/MlData/先做Normalization再做Smote/NCI/sample{i}StandarTrainNci.csv")

