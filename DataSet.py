import numpy
import matplotlib.pyplot as plt
import csv

class DataSet(object):
    
    fused = numpy.empty((0,0))
    gyro = numpy.empty((0,0))
    acc  = numpy.empty((0,0))
    targets = numpy.empty((0,0))
    
    def __init__(self, fused, gyro, acc, targets):
        self.fused = fused
        self.gyro = gyro
        self.acc = acc 
        self.targets = targets
        
        
    @classmethod
    def createFromFile(cls, fileName):
        reader= csv.reader(open('C:\Users\Steve\Documents\Eclipse Projects\BA_Analysis\\dataSets\\'+fileName,"rb"),delimiter=';')
        x=list(reader)
        data = numpy.array(x[1:]).astype('float')
        fused = data[:,0:3]
        gyro = data[:,3:6]
        acc = data[:,6:9]
        targets = data[:,9:]
        return cls(fused,gyro,acc,targets)

       
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
        
    def writeToFile(self, fileName):
        data = numpy.concatenate((self.fused,self.gyro,self.acc,self.targets),1)
        numpy.savetxt("C:\Users\Steve\Documents\Eclipse Projects\BA_Analysis\\dataSets\\"+fileName+".csv", data, delimiter=";")
        
    