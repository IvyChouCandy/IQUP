import numpy as np
import pandas as pd
import csv
import statistics

class analysis:
    def foundPeptide(self,dataDf):
        dataDf["Protein_Peptide"] = dataDf["Protein"] + '_' + dataDf["Peptide"]
        duplicates = dataDf[dataDf.duplicated(subset='Protein_Peptide',keep=False)]
        return duplicates

    def checkTrainandTestPeptide(self,trainDf,testDf):
        trainPPlist = trainDf["Protein_Peptide"].values.tolist()
        testPPlist = testDf["Protein_Peptide"].values.tolist()
        PPintersection = list(set(trainPPlist).intersection(set(testPPlist)))
        for PP in PPintersection:
            testDf = testDf.drop(testDf[testDf["Protein_Peptide"] == PP].index)
        return testDf

    def calculPeptideRatio(self,dataDf,dataEvalue):
        peptideSet = set(dataDf["Protein_Peptide"].values.tolist())
        peptideAreDict = {}
        for peptide in peptideSet:
            pepMedRatioList = []
            fliter = (dataDf["Protein_Peptide"] == peptide)
            analyDataDf = dataDf[fliter]
            for i in range(len(dataEvalue)):
                ratioList = analyDataDf[f"Ratio{i+1}"].values.tolist()
                medRatio = np.median(ratioList)
                medRatio -= dataEvalue[i]
                areRatio = abs(medRatio)/dataEvalue[i]
                pepMedRatioList.append(areRatio)
            are = np.mean(pepMedRatioList)
            peptideAreDict[peptide] = are
        return peptideAreDict

    def calculAllRatio(self, dataDf, dataEvalue):
        pictureAreList = []
        for index,row in dataDf.iterrows():
            rowRatioList = []
            for i in range(len(dataEvalue)):
                ratioCount = row[f"Ratio{i+1}"]
                ratioCount -= dataEvalue[i]
                pratio = abs(ratioCount)/dataEvalue[i]
                rowRatioList.append(pratio)
            are = np.mean(rowRatioList)
            pictureAreList.append(are)
        return pictureAreList


    def writeToCSV(self,dataDict,filename):
        dataDf = pd.DataFrame(list(dataDict.items()),columns=["ProteinPeptide","ARE"])
        dataDf.to_csv(filename,index=False)

    def pictureRatioMain(self,prePath,oriPath,oriYlabelFilename,oriFilename,RatioList,dataEvalueDictList,w_path):
        oriYlabelDf = pd.read_csv(f"{prePath}{oriYlabelFilename}", usecols=["Name", "predict"])
        oriDf = pd.read_csv(f"{oriPath}{oriFilename}", usecols=RatioList)
        oriDf = pd.merge(oriDf, oriYlabelDf, on="Name")
        QupQrpAreDict = {}
        qupFliter = (oriDf["predict"] == 1)
        oriQUPList = self.calculAllRatio(oriDf[qupFliter], dataEvalueDictList)
        qrpFliter = (oriDf["predict"] == 0)
        oriQRPList = self.calculAllRatio(oriDf[qrpFliter], dataEvalueDictList)
        QupQrpAreDict["ARE of QUP"] = oriQUPList
        QupQrpAreDict["ARE of QRP"] = oriQRPList

        with open(f'{w_path}allPictureARE.csv', 'w', newline='') as csvfile:
            fieldnames = ['ARE of QUP', 'ARE of QRP']  # CSV欄位名稱
            writer = csv.writer(csvfile)

            # 寫入欄位名稱
            writer.writerow(fieldnames)

            # 將dictionary的值寫入CSV檔案
            for i in range(max(len(QupQrpAreDict['ARE of QUP']), len(QupQrpAreDict['ARE of QRP']))):
                row = [QupQrpAreDict['ARE of QUP'][i] if i < len(QupQrpAreDict['ARE of QUP']) else '',
                       QupQrpAreDict['ARE of QRP'][i] if i < len(QupQrpAreDict['ARE of QRP']) else '']
                writer.writerow(row)

    def trainandtestare(self,filedf,dataEvalueDictList):
        qupFilter = (filedf["y"] == 1)
        qrpFilter = (filedf["y"] == 0)
        QUPList = self.calculAllRatio(filedf[qupFilter], dataEvalueDictList)
        QRPList = self.calculAllRatio(filedf[qrpFilter], dataEvalueDictList)
        print("QUP:",statistics.mean(QUPList))
        print("QRP:",statistics.mean(QRPList))

filedata = "Lina"   #

prePath = "D:/IQUPexperiment/data/predictAns/"
ansPath = f"D:/IQUPexperiment/MlData/先做Normalization再做Smote/{filedata}/"
oriPath = "D:/IQUPexperiment/originalDataSet/"
w_path = f"D:/IQUPexperiment/PipetideAnalysis/{filedata}/"

oriFilename = f'Classifyout{filedata}.csv'
oriYlabelFilename = f'ori{filedata}.csv'
trainFilename = f"OriTrain{filedata}.csv"
preTestFilename = f"PredictStandarTrain{filedata}StandarTest{filedata}.csv"

#計算ARE
dataEvalueDict = {"Basel":[1,1,1,1,1],"Lina":[2,2,1,1,2,2,1],"NCI":[1,1,0.5,1,1,0.5,1,1,0.5]}
baselRatioList = ["Name","Protein","Peptide","Ratio1","Ratio2","Ratio3","Ratio4","Ratio5"]
linaRatioList = ["Name","Protein","Peptide","Ratio1","Ratio2","Ratio3","Ratio4","Ratio5","Ratio6","Ratio7"]
nciRatioList = ["Name","Protein","Peptide","Ratio1","Ratio2","Ratio3","Ratio4","Ratio5","Ratio6","Ratio7","Ratio8","Ratio9"]

analysisObj = analysis()

# # #first:計算全部圖譜的ARE
# analysisObj.pictureRatioMain(prePath,oriPath,preTestFilename,oriFilename,linaRatioList,dataEvalueDict[filedata],w_path)
#
#second:計算peptide的ARE
oritrainDf = pd.read_csv(f"{ansPath}{trainFilename}", usecols=["Name", "y"])
oritestDf = pd.read_csv(f"{prePath}{preTestFilename}",usecols=["Name","y"]) #,"predict"
oriDf = pd.read_csv(f"{oriPath}{oriFilename}", usecols=linaRatioList)   #

# 比對在原始檔案中與train、test分別相同的Name並取得相關資料
trainNameList = oritrainDf["Name"].values.tolist()
testNameList = oritestDf['Name'].values.tolist()
testDf = oriDf[oriDf["Name"].isin(testNameList)]
trainDf = oriDf[oriDf["Name"].isin(trainNameList)]
testDf = pd.merge(testDf,oritestDf,on ="Name")
trainDf = pd.merge(trainDf,oritrainDf,on = "Name")

#計算train&test的平均圖譜ARE
analysisObj.trainandtestare(trainDf,dataEvalueDict[filedata])
analysisObj.trainandtestare(testDf,dataEvalueDict[filedata])

#
# #找出相同peptide與protein的圖譜並刪除test檔裡在兩個檔案中都存在的圖譜
# #先找出兩個檔案中有2張以上相同protein的peptide圖譜
# duplicateTrainDf = analysisObj.foundPeptide(trainDf)
# duplicateTestDf = analysisObj.foundPeptide(testDf)
# #比對train與test相同的protein的peptide
# cleanTestDf = analysisObj.checkTrainandTestPeptide(duplicateTrainDf,duplicateTestDf)
# cleanTestDf.to_csv(f"{w_path}proteinpeptide.csv")

# #計算在predict中protein_peptide的所有圖譜都被predict成QUP
# allQupPeptideName = []  #被predict成QUP的protein&peptide名稱
# peptideSet = set(cleanTestDf["Protein_Peptide"].values.tolist())
# for peptide in peptideSet:
#     fliter = (cleanTestDf["Protein_Peptide"] == peptide)
#     analyDataDf = cleanTestDf[fliter]
#     labelList = analyDataDf["predict"].values.tolist()
#     if sum(labelList) == len(labelList):    #全部圖譜被predict成QUP
#         allQupPeptideName.append(peptide)
# allQupDf = cleanTestDf[[True if i in allQupPeptideName else False for i in cleanTestDf["Protein_Peptide"]]]
# allQupDf.to_csv(f"{w_path}allQUPInfo.csv")
# allQupDict = analysisObj.calculPeptideRatio(allQupDf,dataEvalueDict[filedata])
# allQupDf = pd.DataFrame(list(allQupDict.items()),columns=["Protein_Peptide","ARE of Peptide"])
# allQupDf.to_csv(f"{w_path}allQUPARE.csv")

# cleanTestDf = cleanTestDf[[True if i not in allQupPeptideName else False for i in cleanTestDf["Protein_Peptide"]]]

#test原本的ARE
# allTestAreDict = analysisObj.calculPeptideRatio(cleanTestDf,dataEvalueDict[filedata]) #
# allTestAreDf = pd.DataFrame(list(allTestAreDict.items()),columns=["Protein_Peptide","ARE of Peptide"])
# allTestAreDf.to_csv(f"{w_path}allTestARE.csv")
# #test檔去掉QUP的ARE
# qrpFilter = (cleanTestDf['predict'] == 0)
# preQrpAreDict = analysisObj.calculPeptideRatio(cleanTestDf[qrpFilter],dataEvalueDict["NCI"])   #
# preQrpAreDf = pd.DataFrame(list(preQrpAreDict.items()),columns=["Protein_Peptide","ARE of Peptide"])
# preQrpAreDf.to_csv(f"{w_path}preQRPARE.csv")

#分析are>0.25的peptide
# alltestDf = pd.read_csv("D:/IQUPexperiment/PipetideAnalysis/NCI/allTestARE.csv")
# preQRPDf = pd.read_csv("D:/IQUPexperiment/PipetideAnalysis/NCI/preQRPARE.csv")
# proteinDf = pd.read_csv("D:/IQUPexperiment/PipetideAnalysis/NCI/proteinpeptide.csv")
# testarefilter = (alltestDf["ARE of Peptide"] >= 0.25)
# testDf = alltestDf[testarefilter]
# prearefilter = (preQRPDf["ARE of Peptide"] >= 0.25)
# preDf = preQRPDf[prearefilter]
# testSet = set(testDf["Protein_Peptide"].values.tolist())
# preSet = set(preDf["Protein_Peptide"].values.tolist())
# proteinSet = testSet|preSet
# areProteinDf = proteinDf[[True if i in proteinSet else False for i in proteinDf["Protein_Peptide"]]]
# areProteinDf.to_csv("D:/IQUPexperiment/PipetideAnalysis/NCI/are大於0.25之分析.csv")

# #算區間的postiverate
# file = "NCI"
# prob_path = f"D:/IQUPexperiment/data/0614分析/{file}/"
# answer_path = f"D:/IQUPexperiment/MlData/先做Normalization再做Smote/{file}/"
# modelNameList = ['lightgbm', 'catboost', 'gbc', 'nb', 'et', 'rf', 'qda',
#                  'xgboost']
# analysisDict = {}
#
# bins = [-np.inf,0.3,0.5,0.7,np.inf]
# labels = ['<0.3', '0.3-0.5', '0.5-0.7', '>0.7']
#
# for model in modelNameList:
#     modelDict = {}
#     probVectorDf = pd.read_csv(f"{prob_path}{model}predictAnswer.csv")
#     if "y" in probVectorDf.columns:
#         probVectorDf = probVectorDf.drop(["y"],axis=1)
#     answerDf = pd.read_csv(f"{answer_path}OriTest{file}.csv",usecols=["Name","y"])
#     probVectorAnswerDf = pd.merge(probVectorDf,answerDf,on="Name",how='inner')
#     probVectorAnswerDf[f"{model}_interval"] = pd.cut(probVectorAnswerDf[f"{model}"], bins=bins, labels=labels, include_lowest=True)
#     interval_counts = probVectorAnswerDf[f'{model}_interval'].value_counts().sort_index()
#     matched_counts = probVectorAnswerDf[
#         (probVectorAnswerDf[f'{model}predict'] == 1) &
#         (probVectorAnswerDf['y'] == 1)
#         ][f'{model}_interval'].value_counts().sort_index()
#     matched_counts = matched_counts.reindex(labels, fill_value=0)
#     ratios = (matched_counts / interval_counts).fillna(0)
#     result_dict = {
#         interval: {
#             'total': int(interval_counts[interval]),
#             'matched': int(matched_counts[interval]),
#             'ratio': float(ratios[interval])
#         }
#         for interval in labels
#     }
#     analysisDict[model] = result_dict
# data = []
# for model, intervals in analysisDict.items():
#     for interval, stats in intervals.items():
#         data.append({
#             'Model': model,
#             'Interval': interval,
#             'Total': stats['total'],
#             'Matched': stats['matched'],
#             'Ratio': stats['ratio']
#         })
# analysisdf = pd.DataFrame(data)
# analysisdf = analysisdf[['Model', 'Interval', 'Total', 'Matched', 'Ratio']]
# analysisdf.to_csv(f'{prob_path}model_comparison.csv', index=False)
