'''
Created on 09.03.2016

@author: Steve
'''
import numpy as np
import matplotlib.pyplot as plt
import DataSet
from sklearn.manifold.spectral_embedding_ import SpectralEmbedding
from mpl_toolkits.mplot3d import Axes3D





def plot3DData(X):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0], X[:,1],X[:,2], marker='o')
    
    
def plot(X, title=''):
    plt.figure()
    plt.title(title)
    plt.plot(X)
    
def subPlot(X, title = [], lim =[-2,2]):
    nInputFiles = len(X)
    _, axes = plt.subplots(nInputFiles, 1)
    for row, i in zip(axes,range(0,nInputFiles)):
        if(i<len(title)):
            row.set_title(title[i])
        row.plot(X[i])
        row.set_ylim(lim)  
        
def subBoxPlot(X, titles = [],title=''):
    yLim = 0
    for set in X:
        absVals = np.absolute(set)
        newLim = np.max(absVals.flatten())
        if newLim > yLim:
            yLim = newLim
    
    nInputFiles = len(X)
    fig, axes = plt.subplots(nInputFiles, 1)
    plt.title(title)
    for row, i in zip(axes,range(0,nInputFiles)):
        row.set_title(titles[i])
        row.boxplot(X[i])
        row.set_ylim([-yLim,yLim])    



    
if __name__ == '__main__':
    #inputFiles = ['nadja_0_1.npz', 'nadja_0_2.npz', 'nadja_0_3.npz']
    inputFiles = ['stephan_0_0.npz', 'stephan_0_1.npz', 'stephan_0_2.npz']
    randTestFiles = ['daniel_0_0.npz','stephan_0_0.npz','stephan_1_2.npz']
    
   
    dataSets = []
    dataSetMeans = []
    dataSetSums = []
    for iFile in inputFiles:
        set = DataSet.createDataSetFromFile(iFile)
        dataSets.append(set)
        allSignals = set.getAllSignals(0)
        means = np.zeros((len(allSignals),len(allSignals[0][0])))
        sums = np.zeros((len(allSignals),len(allSignals[0][0]-1)))
        for signal, i in zip(allSignals,range(0,len(allSignals))):
            means[i,:] = np.mean(signal, 0)
            sums[i,:] = np.sum(signal, 0)
        print np.mean(means, 0)
        dataSetMeans.append(means)
        dataSetSums.append(sums[:,0:9])
    subBoxPlot(dataSetMeans, inputFiles)
    subBoxPlot(dataSetSums, inputFiles)        
        
        


        

def specEmbedding():
    clf = SpectralEmbedding(3)
    X = clf.fit_transform(dataSets[2].getDataForTraining([0], 2)[0][:])
    c = range(0,len(X))
    plt.figure()
    plt.plot(X[:,0], X[:,1])
    plt.scatter(X[:,0], X[:,1], c=c, cmap='gray')
    

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X[:,0], X[:,1],X[:,2])
    ax.scatter(X[:,0], X[:,1],X[:,2], c=c, marker='o')
    
    
    
def plotMatrix(w,treshold = 0):
    mat = np.copy(w)
    mat[np.abs(w) < 0.12] = 0
    plt.figure() 
    plt.imshow(mat, cmap="hot")
    plt.colorbar()
    

    