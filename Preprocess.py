
import numpy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy
import mdp
import csv
from thread import start_new_thread
import DataSet
from DataAnalysis import plot
from Main import getProjectPath

def readFileToNumpy(fileName):
    reader=csv.reader(open(fileName,"rb"),delimiter=',')
    x=list(reader)
    return numpy.array(x[1:]).astype('float')

def separateInputData(fileData):
    fused = numpy.atleast_2d(fileData[:,1:4])
    gyro = numpy.atleast_2d(fileData[:,4:7])
    acc = numpy.atleast_2d(fileData[:,7:10])
    targets = numpy.atleast_2d(fileData[:,10:])
    return fused, gyro, acc, targets




def transformToDelta(vals):
    newVals = numpy.zeros((len(vals),len(vals[0])))
    for i in range(1,len(vals)):
        newVals[i-1] = vals[i]-vals[i-1]
    return newVals

def removeLOverflow(fused):
    for i in range(0,len(fused)):
        if fused[i,0] > numpy.pi:
            fused[i,0] = fused[i,0] - 2*numpy.pi
        if fused[i,0] < -numpy.pi:
            fused[i,0] = fused[i,0] + 2*numpy.pi
            
        if fused[i,1] > numpy.pi:
            fused[i,1] = fused[i,1] - 2*numpy.pi
        if fused[i,1] < -numpy.pi:
            fused[i,1] = fused[i,1] + 2*numpy.pi

        if fused[i,2] > numpy.pi:
            fused[i,2] = fused[i,2] - 2*numpy.pi
        if fused[i,2] < -numpy.pi:
            fused[i,2] = fused[i,2] + 2*numpy.pi
    return fused

def applyActivationFilter(inputData, width):
    actLevel = numpy.sum(numpy.abs(inputData),1)
    target = numpy.zeros((len(inputData),1))
    for i in range(width,len(inputData-width)):
            target[i] = numpy.mean(actLevel[i-width:i+width])
    return target



def centerAndNormalize(inputData):
    means = numpy.mean(inputData, 0)
    centered = inputData - means
    vars = numpy.std(centered, 0)
    normalized = centered/vars
    return normalized, means, vars


def getTrainingBeginAndEndIndex(targetSig):
    beginInd = 0
    endInd = len(targetSig)
    for i in range(0,len(targetSig)):
            if targetSig[i] == 1:
                beginInd= i-1;
                break
    for i in range(0,len(targetSig)):
            if targetSig[len(targetSig)-1-i] == 1:
                endInd= len(targetSig)-i;
                break
    return beginInd,endInd


def formatDataSet(data):
    print data.shape
    newStart = input("Start:")
    newEnd = input("End:")
    newData = data[newStart:newEnd,:]
    return newData

def formatTargetFilter(data):
    treshold = input('Treshold:')
    targetFunction = applyFormatTargetFilter(data, treshold)
    plt.figure()
    plt.plot(data[:,9])
    plt.plot(data[:,10])
    plt.plot(targetFunction)
    return targetFunction

def applyFormatTargetFilter(data, treshold):
    targetFunction = (data[:,10] > treshold).astype(float)
    return numpy.atleast_2d(targetFunction).T
    
def removeArea(data):
    cutOutStart = input("Start:")
    cutOutEnd = input("End:")
    newDataStart = data[:cutOutStart,:]
    newDataEnd = data[cutOutEnd:,:]
    return numpy.concatenate((newDataStart,newDataEnd))
    
    
def plotData(data):
        plt.figure()
        plt.clf()
        plt.subplot(411)
        plt.title('Fused')
        plt.plot(data[:,0:3])
        plt.plot(data[:,9])
        plt.plot(data[:,10])
        
        plt.subplot(412)
        plt.title('Gyro')
        plt.plot(data[:,3:6])
        plt.plot(data[:,9])
        plt.plot(data[:,10])
        
        plt.subplot(413)
        plt.title('Acc')
        plt.plot(data[:,6:9])
        plt.plot(data[:,9])
        plt.plot(data[:,10])
        
        
        plt.subplot(414)
        plt.title('Targets')
        plt.plot(data[:,9])
        plt.plot(data[:,10])  
        plt.show()
    
    
def writeToCSV(data,fileName):
    numpy.savetxt(getProjectPath()+"\\dataSets\\"+fileName+".csv", data, delimiter=";")
    
def safeToDataSet(fileName, data, means, stds, gestures, targetTreshold):
    ds = DataSet.DataSet(data[:,0:3],data[:,3:6],data[:,6:9],numpy.append(data[:,9:], applyFormatTargetFilter(data, targetTreshold), 1), \
                    means, stds, gestures)
    ds.writeToFile(fileName)
    
    
def load(nr):
    global i
    plt.close('all')
    i = readFile("nadja\\nadja_"+str(nr)+".csv")
    plotData(i)

    
def safe(inputData,aaa,nr):
    writeToCSV(numpy.concatenate((inputData,numpy.atleast_2d(aaa).T),1),"nadja_fitted_"+str(nr))

def readFile(fileName):
    return readFileToNumpy(getProjectPath()+'dataSets\\'+fileName)

if __name__ == '__main__':
    
#def main():
    


    inputFileName = ["2016-03-19-11-26-58-nadja_fullSet.csv"]
    
    fileData = numpy.zeros((1,31))
    for fileName in inputFileName:
        newData = readFileToNumpy(getProjectPath()+'dataSets\\'+fileName)
        fileData = numpy.append(fileData,newData,0)
    
    fused, gyro, acc, targets = separateInputData(fileData)


    fused = transformToDelta(fused)
    fused = removeLOverflow(fused)
    
    _, f_means, f_stds = centerAndNormalize(fused)
    _, g_means, g_stds = centerAndNormalize(gyro)
    _, a_means, a_stds = centerAndNormalize(acc)
    
    means = numpy.concatenate((f_means,g_means,a_means),0)
    stds = numpy.concatenate((f_stds,g_stds,a_stds),0)
    gestures = numpy.max(targets,0)
    
    dataSets = []
    gestureSets = []
    for i in range(0,len(targets[0])):
        start, end = getTrainingBeginAndEndIndex(targets[:,i])
        t_fused = fused[start:end,:]
        t_gyro = gyro[start:end,:]
        t_acc = acc[start:end,:]
        t_target =numpy.atleast_2d(targets[start:end,i]).T
        t_accFilter = applyActivationFilter(numpy.concatenate((t_fused,t_gyro,t_acc),1),6)
        a = numpy.concatenate((t_fused,t_gyro,t_acc,t_target,t_accFilter),1)
        dataSets.append(a)
        gestureSets.append(numpy.max(targets[start:end,:],0))
    plotData(dataSets[0])
    plotData(dataSets[1])    
    plotData(dataSets[2])    
    plotData(dataSets[3])    
    plotData(dataSets[4])    
    plotData(dataSets[5])    
    