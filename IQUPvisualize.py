import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import umap
import os
import time

class Visualization:
    def __init__(self, dataFile, outputPrefix, outputDir, colors=['red', 'green'], sizes=[5, 5],marker = ['o','.']):
        self.dataFile = dataFile
        self.outputName = outputPrefix
        self.outputPath = outputDir
        self.colors = colors
        self.sizes = sizes
        self.outputTsnePath = f'{outputDir}/tsne/'
        self.outputUmapPath = f'{outputDir}/umap/'
        self.marker = marker
        if not os.path.isdir(self.outputPath):
            os.mkdir(self.outputPath)
        if not os.path.isdir(self.outputTsnePath):
            os.mkdir(self.outputTsnePath)
        if not os.path.isdir(self.outputUmapPath):
            os.mkdir(self.outputUmapPath)


    def plotPca(self,trainFeatureList):
        # 讀取數據
        df = pd.read_csv(self.dataFile, usecols=trainFeatureList)
        y = df['y']
        X = df.drop(["y"], axis=1)

        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        # 繪製PCA圖
        fig, ax = plt.subplots(figsize=(8, 6))
        for i in range(len(self.colors)):
            ax.scatter(X_pca[y == i, 0], X_pca[y == i, 1], c=self.colors[i], s=self.sizes[i],marker = self.marker[i],
                       label='Negatives' if i == 0 else 'Positives')
        ax.set_xlabel('PC1', fontsize=20)
        ax.set_ylabel('PC2', fontsize=20)
        ax.tick_params(axis='both', labelsize=18)
        ax.legend(fontsize=14)

        # 存檔
        plt.savefig(self.outputPath + self.outputName + '_PCA.png', dpi=300)

    def plotTsne(self, perplexityList, trainFeatureList):
        # 讀取數據
        df = pd.read_csv(self.dataFile, usecols=trainFeatureList)
        y = df['y']
        X = df.drop(["y"], axis=1)

        # t-SNE
        randomInt = np.random.randint(0, 100)
        for perplexity in perplexityList:
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=randomInt)
            X_tsne = tsne.fit_transform(X)

            # 繪製t-SNE圖
            fig, ax = plt.subplots(figsize=(8, 6))
            for i in range(len(self.colors)):
                ax.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], c='none', s=self.sizes[i],marker = self.marker[i],
                           edgecolors=self.colors[i],label='Negatives' if i == 0 else 'Positives')
            ax.set_xlabel('Dimension1', fontsize=25)
            ax.set_ylabel('Dimension2', fontsize=25)
            ax.tick_params(axis='both', labelsize=23)
            ax.legend(fontsize=19)
            plt.tight_layout()

            # 存檔
            plt.savefig(self.outputTsnePath + self.outputName + f'_t-SNE_{randomInt}_perplexity={perplexity}.png', dpi=300)

    def runUmap(self,n_neighbors_list,min_dist_list,trainFeatureList):  #執行umap
        self.df = pd.read_csv(self.dataFile, usecols=trainFeatureList)
        for n_neighbors in n_neighbors_list:
            for min_dist in min_dist_list:
                self.umap_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2)#,random_state=2023
                self.umap_result =self.umap_model.fit_transform(self.df)#使用fit試試看
                self.plotUmap(n_neighbors=n_neighbors, min_dist=min_dist)  #umap繪圖
        return

    def plotUmap(self, n_neighbors, min_dist): #umap繪圖
        umapResultFor0List = []
        umapResultFor1List = []
        for lableNum in range(len(self.df['y'])):
            if self.df['y'][lableNum] == 0:
                umapResultFor0List.append(self.umap_result[lableNum])
            elif self.df['y'][lableNum] == 1:
                umapResultFor1List.append(self.umap_result[lableNum])
            else:
                print("ERROR")
        umap_result_for0_array = np.array(umapResultFor0List)
        umap_result_for1_array = np.array(umapResultFor1List)

        nonaip = plt.scatter(umap_result_for0_array[:, 0], umap_result_for0_array[:, 1], c='none', s=self.sizes[0], marker=self.marker[0],
                             edgecolors=self.colors[0], linewidths=0.5)
        aip = plt.scatter(umap_result_for1_array[:, 0], umap_result_for1_array[:, 1], c='none', s=self.sizes[1], marker=self.marker[1],
                          edgecolors=self.colors[1])
        plt.legend([nonaip, aip], ['non-AIP', 'AIP'])
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlabel('Dimension1', fontsize=18)
        plt.ylabel('Dimension2', fontsize=18)
        plt.gca().set_aspect('equal', 'datalim')
        plt.title(self.outputName + ' n_neighbors=' + str(n_neighbors) + ' min_dist=' + str(min_dist) + '\n',
                  fontsize=18)
        plt.savefig(f"{self.outputUmapPath}{self.outputName}_{self.outputName } n_neighbors={str(n_neighbors)} min_dist={str(min_dist)}.png")
        plt.savefig(
            f"{self.outputUmapPath}{self.outputName}n_neighbors={str(n_neighbors)} min_dist={str(min_dist)}.png")
        return

trainFeatureList = ['Mass','PTMCnt','PTMCntRatio','MassDifferent','PeptideLength','AvgIntensityLog','expect','fval','EucliIntraProtein','ManhattanIntraProtein','PearsonIntraProtein','CosineIntraProtein','EucliIntraProteinPair','ManhattanIntraProteinPair','CosineIntraProteinPair','Charge2','Charge3','Charge4','Charge5','y']

# outputPath = "../data/Visualization/"
# vis = Visualization(data_file='train_F150_DS1.csv', output_name='F150_DS1', output_dir='./data/',
#                     colors=['blue', 'red'], sizes=[5, 5])
# for i in range(5):
visObj = Visualization(dataFile=f"D:/IQUPexperiment/MlData/先做Normalization再做Smote/Lina/StandarTrainLina.csv",
                       outputPrefix=f'StandarTrainLina',
                       outputDir="./data/Visualization/", colors=['blue', 'red'], sizes=[2, 2],marker = ['o','.'])#size調小(原本為5) 圓變成空心的 NCI要更小

# # 繪製PCA圖
# visObj.plotPca(trainFeatureList)

# 繪製t-SNE圖
perplexityList = [50, 65,80, 100]  #2, 3, 5, 7, 10, 15, 30, 40,
visObj.plotTsne(perplexityList, trainFeatureList)


# # UMAP
# nNeighborsList = [2, 3, 5, 7, 10, 15, 30, 40, 50, 65, 80, 100]
# minDistList = [0, 0.01, 0.025, 0.05, 0.1, 0.25, 0.35, 0.5, 0.6, 0.75, 0.85, 0.99]  # 可視情況更改
# visObj.runUmap(nNeighborsList, minDistList, trainFeatureList)  # 改dataSetNameStr檔名




