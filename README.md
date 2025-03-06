# IQUP
IQUP: Identification of quantitatively unreliable spectra with machine learning for isobaric labeling-based proteomics

# Description
This is the source code of IQUP, a machine learning predictor for identifying quantitatively unreliable spectra for isobaric labeling-based proteomics. The trained models are also included in this package, allowing predictions on a given isobaric labeling proteomic dataset.

# Installation
Requiremenets:
* Python = 3.8, pycaret[full] = 2.3.10

Packages
* Install required packages using `pip install -r requirements.txt`

# Usage
Modify IQUPdataPrep.py for your data set in csv format

* Input file
  * One or more files in csv format (described below)

* output file
  * StandarTrain{filename}.csv/StandarTest{filename}.csv -- Contains the column information from Column to analyze.txt.
    
   ![image](https://github.com/IvyChouCandy/IQUP/blob/main/dataColumn.jpg)

Set the path

```py
# Path setting
Datapath = "../MlData/{foldername}/{csvfilename}/"  #Your preprocessed data will be saved in this path 
tuneModelPath = "../data/tuneModel"  #This path by default consists of ML models
finalModelPath = "../data/finalModel"  #This path by default consists of ML models
scorePath = "../data/Score/"  # Prediction score files will be saved in the path
paramPath = "../data/param/"  #The path by default consists of MLmodels param
w_path = ../data/analysisData/"  #Analysis ARE of peptide  files will be saved in the path
```

Run mainIQUP.py
```py
# If you want to use different model, you can change modelNameList
modelNameList = ['lightgbm', 'catboost', 'gbc', 'nb', 'et', 'rf', 'qda',
                 'xgboost','ridge','lr', 'lda','ada', 'knn', 'mlp','dt','svm','gpc','rbfsvm']
```

Here is the code snippet in mainIQUP.py. We already set the parameters and the program is ready to be excecuted.
```py
tunedModelList, tunerList = pycObj.doTuneModel(searchLibrary='optuna', searchAlg='tpe', includeModelList=modelNameList, foldNum=5,
                       n_iter=10, early_stopping=False, customGridDict=None)
pycObj.doFinalizeModel(tunedModelList)
pycObj.doSaveModel(finalModelPath, b_isFinalizedModel=True)
finalModelList = pycObj.doLoadModel(path=finalModelPath,fileNameList=modelNameList, b_isFinalizedModel=True)
predObjIndp = Predict(dataX=testDataDf_X, modelList=finalModelList)
predVectorListIndp, probVectorListIndp = predObjIndp.doPredict()
```
