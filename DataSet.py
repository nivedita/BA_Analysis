import numpy
import matplotlib.pyplot as plt
import csv
from sklearn.cluster.mean_shift_ import MeanShift


class DataSet(object):
    
    fused = numpy.empty((0,0))
    gyro = numpy.empty((0,0))
    acc  = numpy.empty((0,0))
    targets = numpy.empty((0,0))
    means = numpy.empty((0,0))
    stds = numpy.empty((0,0))
    gestures = numpy.empty((0,0))
    
    
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
        return  numpy.concatenate((self.fused,self.gyro,self.acc,self.targets),1)
    
    def getFused(self):
        return self.fused
    
    def getAcc(self):
        return self.acc
    
    def getGyro(self):
        return self.gyro 
    
    def getDataForTraining(self, classNr,useFused=True, useGyro=True, useAcc=True, targetNr=2, multiplier = 1):
        inputData = numpy.empty((0,0))
        if useFused:
            inputData = self.fused
        
        if useGyro:
            if len(inputData) == 0:
                inputData = self.gyro
            else:
                inputData = numpy.append(inputData, self.gyro, 1)
            
        if useAcc:
            if len(inputData) == 0:
                inputData = self.acc
            else:
                inputData = numpy.append(inputData, self.acc, 1)
        if self.gestures[classNr] == 1:
            readOutTrainingData = numpy.atleast_2d(self.targets[:,targetNr]).T
        else:
            readOutTrainingData = numpy.atleast_2d(numpy.zeros((len(self.targets),1)))
        data = inputData
        target = readOutTrainingData
        for i in range(0,multiplier):
            data = numpy.append(data,inputData,0)
            target = numpy.append(target,readOutTrainingData,0)
        return (data,target)
    
    
    def getMinusPlusDataForTraining(self, classNr ,useFused=True, useGyro=True, useAcc=True, targetNr=2, multiplier = 1):
        inputData, target = self.getDataForTraining(classNr, useFused, useGyro, useAcc, targetNr, multiplier)
        low_values_indices = target == 0  # Where values are low
        target[low_values_indices] = -1   
        return (inputData,target)
        
        
    def writeToFile(self, fileName):
        numpy.savez("C:\Users\Steve\Documents\Eclipse Projects\BA_Analysis\\dataSets\\"+fileName,  \
                    fused=self.fused,gyro=self.gyro,acc=self.acc,targets=self.targets,means=self.means,stds=self.stds,gestures=self.gestures)
        
        
def createDataSetFromFile(fileName):
    data = numpy.load('C:\Users\Steve\Documents\Eclipse Projects\BA_Analysis\\dataSets\\'+fileName)
    fused = data['fused']
    gyro = data['gyro']
    acc = data['acc']
    targets = data['targets']
    means = data['means']
    stds = data['stds']
    gestures = data['gestures']
    return DataSet(fused, gyro, acc, targets,means, stds, gestures)
    