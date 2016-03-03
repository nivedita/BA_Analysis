import numpy
import matplotlib.pyplot as plt
import csv

class DataSet(object):
    
    fused = numpy.empty((0,0))
    gyro = numpy.empty((0,0))
    acc  = numpy.empty((0,0))
    targets = numpy.empty((0,0))
    
    @classmethod
    def createFromFile(cls,self, fused, gyro, acc, targets):
        self.fused = fused
        self.gyro = gyro
        self.acc = acc 
        self.targets = targets
        
        
    def __init__(self, fileName):
        reader= csv.reader(open('C:\Users\Steve\Documents\Eclipse Projects\BA_Analysis\\dataSets\\'+fileName,"rb"),delimiter=';')
        x=list(reader)
        data = numpy.array(x[1:]).astype('float')
        self.fused = data[:,0:3]
        self.gyro = data[:,3:6]
        self.acc = data[:,6:9]
        self.targets = data[:,9:]


       
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
    
    def getDataForTraining(self, useFused=True, useGyro=True, useAcc=True, targetNr=2, multiplier = 1):
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
        readOutTrainingData = numpy.atleast_2d(self.targets[:,targetNr]).T
        
        data = inputData
        target = readOutTrainingData
        for i in range(0,multiplier):
            data = numpy.append(data,inputData,0)
            target = numpy.append(target,readOutTrainingData,0)
        return (data,target)
        
        
        
    def writeToFile(self, fileName):
        data = numpy.concatenate((self.fused,self.gyro,self.acc,self.targets),1)
        numpy.savetxt("C:\Users\Steve\Documents\Eclipse Projects\BA_Analysis\\dataSets\\"+fileName+".csv", data, delimiter=";")
        
    