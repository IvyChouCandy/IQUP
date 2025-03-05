import pandas as pd

oriFileDf = pd.read_csv("D:/IQUPexperiment/originalDataSet/ClassifyoutBasel.csv")
fileDf = pd.read_csv("D:/IQUPexperiment/analysis Data/0801/Taining&TestData所有資料筆數/OriTrainBasel.csv",usecols=["Name"])

nameList = fileDf["Name"].values.tolist()
fileDf = oriFileDf[oriFileDf["Name"].isin(nameList)]

dataEvalueDict = {"Basel":[1,1,1,1,1],"Lina":[2,2,1,1,2,2,1],"NCI":[1,1,0.5,1,1,0.5,1,1,0.5]}

baselRatioList = ["Name","Protein","Peptide","Ratio1","Ratio2","Ratio3","Ratio4","Ratio5"]
# linaRatioList = ["Name","Protein","Peptide","Ratio1","Ratio2","Ratio3","Ratio4","Ratio5","Ratio6","Ratio7"]
# nciRatioList = ["Name","Protein","Peptide","Ratio1","Ratio2","Ratio3","Ratio4","Ratio5","Ratio6","Ratio7","Ratio8","Ratio9"]



