# IQUP
IQUP: Identification of quantitatively unreliable spectra with machine learning for isobaric labeling-based proteomics

# Abstract
Mass spectrometry‑based proteomics using isobaric labeling technology has become popular for proteomic quantitation. Existing approaches rely on the mechanism of target-decoy search and false discovery rate control to examine whether a peptide-spectrum match (PSM) is utilized for quantitation. However, some PSMs passing the examination may still exhibit high quantitation errors, which can deteriorate the overall quantitation accuracy. We present IQUP, a machine learning-based method to identify quantitatively unreliable PSMs, termed QUPs. PSMs were characterized by 16 spectral and distance-based features for machine learning. Independent test results reveal that the best-performing models for the three datasets achieve accuracies of 0.883-0.966, AUCs of 0.924-0.963, and MCCs of 0.596-0.691. Notably, the distribution of relative errors for QUPs and quantitatively reliable PSMs (QRPs) exhibit significant differences. By using only the predicted QRPs for peptide-level quantitation, the proportions of peptides with larger relative errors decrease significantly, with a range between 15.3% and 83.3% for the three datasets; in the meantime, the proportions of peptides with smaller relative errors increase by 3.1%-25.5%. Our experimental results demonstrate that IQUP provides robust performance and strong generalizability across multiple datasets and has great potential in improving proteomic quantitation accuracy at PSM and peptide levels for isobaric labeling experiments.

# Description
We uploaded the source code, datasets, and trained models of IQUP, which is a tool that identifies QUPs, the quantitatively unreliable PSMs (peptide-spectrum match), from isobaric labeling datasets based on TMT or iTRAQ. Users have to provide csv files consisting of a list of validated PSMs and their features (described below) for a specific proteomic dataset. Then, IQUP can be applied to predict the spectra that may have high quantitation errors. 

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
Datapath = "../MlData/{foldername}/{csvfilename}/"  # the preprocessed data will be saved in this path 
tuneModelPath = "../data/tuneModel"  #This path by consists of fine-tuned ML models that are trained with the training set in cross validation
finalModelPath = "../data/finalModel"  #This path consists of fine-tuned ML models that were trained with the entire dataset
scorePath = "../data/Score/"  # Prediction score files will be saved in the path
paramPath = "../data/param/"  #The path consists of ML models' parameters
w_path = ../data/analysisData/"  # The path consists of files for peptides' average relative errors
```

Run mainIQUP.py
```py
# If you want to use different models, you can change modelNameList
modelNameList = ['lightgbm', 'catboost', 'gbc', 'nb', 'et', 'rf', 'qda']
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
