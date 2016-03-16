import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import csv
from sklearn.cluster.mean_shift_ import MeanShift
from Main import getProjectPath


class DataSet(object):
    
    fused = np.empty((0,0))
    gyro = np.empty((0,0))
    acc  = np.empty((0,0))
    targets = np.empty((0,0))
    means = np.empty((0,0))
    stds = np.empty((0,0))
    gestures = np.empty((0,0))
    
    
    def __init__(self, fused, gyro, acc, targets,means, stds, gestures):
        self.fused = fused
        self.gyro = gyro
        self.acc = acc 
        self.targets = targets
        self.means = means
        self.stds = stds
        self.gestures = gestures
        



       
    def plot(self, targetNr=2):
        plt.figure()
        plt.clf()
        plt.subplot(411)
        plt.title('Fused')
        plt.plot(self.fused)
        plt.plot(self.targets[:,targetNr])
        
        
        plt.subplot(412)
        plt.title('Gyro')
        plt.plot(self.gyro)
        plt.plot(self.targets[:,targetNr])

        
        plt.subplot(413)
        plt.title('Acc')
        plt.plot(self.acc)
        plt.plot(self.targets[:,targetNr])
        
        
        plt.subplot(414)
        plt.title('Targets')
        plt.plot(self.targets)
        plt.show()
        
    def getData(self):
        return  np.concatenate((self.fused,self.gyro,self.acc,self.targets),1)
    
    def getFused(self):
        return self.fused
    
    def getAcc(self):
        return self.acc
    
    def getGyro(self):
        return self.gyro 
    
    def getDataForTraining(self, classNrs,useFused=True, useGyro=True, useAcc=True, targetNr=2, multiplier = 1, normalized = False):
        inputData = np.empty((0,0))
        stds = np.empty((0,0))
        if useFused:
            inputData = self.fused
            stds = self.stds[0:3]
        if useGyro:
            if len(inputData) == 0:
                inputData = self.gyro
                stds = self.stds[3:6]
            else:
                inputData = np.append(inputData, self.gyro, 1)
                stds = np.append(stds,self.stds[3:6],0)
            
        if useAcc:
            if len(inputData) == 0:
                inputData = self.acc
                stds = self.stds[6:9]
            else:
                inputData = np.append(inputData, self.acc, 1)
                stds = np.append(stds,self.stds[6:9],0)
        
        readOutTrainingData = np.zeros((len(inputData),len(classNrs)))
        for classNr in classNrs:
            readOutTrainingData[:,classNr] = self.targets[:,targetNr].T * self.gestures[classNr]
        
        if normalized:
            inputData = inputData/stds
        data = inputData
        target = readOutTrainingData
        for i in range(0,multiplier):
            data = np.append(data,inputData,0)
            target = np.append(target,readOutTrainingData,0)
        return (data,target)
    
    def getAllSignals(self, gesture, targetNr = 2):
        signals = []
        target = self.targets[:,targetNr]
        if self.gestures[gesture] != 0:
            changesT = np.where(target[:-1] != target[1:])[0] + 1
            lastInd = 0
            for ind in changesT:
                if target[lastInd] == 1:
                    signals.append(np.concatenate((self.fused[lastInd:ind,:],self.gyro[lastInd:ind,:],self.acc[lastInd:ind,:],np.atleast_2d(target[lastInd:ind]).T),1))
                lastInd = ind
        return signals
    
    def getMinusPlusDataForTraining(self, classNr ,useFused=True, useGyro=True, useAcc=True, targetNr=2, multiplier = 1):
        inputData, target = self.getDataForTraining(classNr, useFused, useGyro, useAcc, targetNr, multiplier, True)
        low_values_indices = target == 0  # Where values are low
        target[low_values_indices] = -1   
        return (inputData,target)
        
        
    def unnormalize(self):
        self.fused = np.add(np.multiply(self.fused,self.stds[0:3]),self.means[0:3])
        self.gyro = np.add(np.multiply(self.gyro,self.stds[3:6]),self.means[3:6])
        self.acc = np.add(np.multiply(self.acc,self.stds[6:9]),self.means[6:9])
        
    def writeToFile(self, fileName):
        np.savez(getProjectPath()+'dataSets/'+fileName,  \
                    fused=self.fused,gyro=self.gyro,acc=self.acc,targets=self.targets,means=self.means,stds=self.stds,gestures=self.gestures)
        
        
def createDataSetFromFile(fileName):
    data = np.load(getProjectPath()+'dataSets/'+fileName)
    fused = data['fused']
    gyro = data['gyro']
    acc = data['acc']
    targets = data['targets']
    means = data['means']
    stds = data['stds']
    gestures = data['gestures']
    return DataSet(fused, gyro, acc, targets,means, stds, gestures)


def appendDS(dataSets, usedGestures, useFused=True, useGyro=True, useAcc=True):
    result = (dataSets[0].getDataForTraining(usedGestures,useFused, useGyro, useAcc, 2)[0],\
              dataSets[0].getDataForTraining(usedGestures,useFused, useGyro, useAcc, 2)[1])
    for i in range(1,len(dataSets)):
        result = (np.append(result[0], \
                  dataSets[i].getDataForTraining(usedGestures,useFused, useGyro, useAcc, 2)[0],0), \
                  np.append(result[1], \
                  dataSets[i].getDataForTraining(usedGestures,useFused, useGyro, useAcc, 2)[1],0))
    return result

#def appendDataSets(ds1, ds2):
#    fused = np.append(ds1.fused, ds2.fused, 0)
#    gyro = np.append(ds1.gyro, ds2.gyro, 0)
#    acc =  np.append(ds1.acc, ds2.acc, 0)
#    targets = np.append(ds1.targets, ds2.targets, 0)
#    gestures = np.max(np.append(np.atleast_2d(ds1.gestures),np.atleast_2d(ds2.gestures),0),0)
#    return DataSet(fused, gyro, acc, targets,[], [], gestures)
    