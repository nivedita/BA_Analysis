import numpy
import matplotlib as mpl
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import scipy
import mdp
import csv
import Oger
import datetime
from matplotlib.backends.backend_pdf import PdfPages

import DataSet
from Evaluation import * 
import Evaluation
from DataAnalysis import plot
from DataAnalysis import subPlot
from DataSet import createDataSetFromFile, createData, appendDS
from Main import getProjectPath


def getAllDataSetNames():
    julianFiles = ['julian_0_fullSet.npz','julian_1_fullSet.npz','julian_2_fullSet.npz','julian_3_fullSet.npz','julian_4_fullSet.npz','julian_5_fullSet.npz','julian_8_fullSet.npz','julian_9_fullSet.npz']
    nikeFiles = ['nike_0_fullSet.npz','nike_1_fullSet.npz','nike_2_fullSet.npz','nike_3_fullSet.npz','nike_4_fullSet.npz','nike_5_fullSet.npz','nike_8_fullSet.npz','nike_9_fullSet.npz']
    stephanFiles = ['stephan_0_fullSet.npz','stephan_1_fullSet.npz','stephan_2_fullSet.npz','stephan_3_fullSet.npz','stephan_4_fullSet.npz','stephan_5_fullSet.npz','stephan_8_fullSet.npz','stephan_9_fullSet.npz']
    nadjaFiles = ['nadja_0_fullSet.npz','nadja_1_fullSet.npz','nadja_2_fullSet.npz','nadja_3_fullSet.npz','nadja_4_fullSet.npz','nadja_5_fullSet.npz','nadja_8_fullSet.npz','nadja_9_fullSet.npz']
    lineFiles = ['line_0_fullSet.npz','line_1_fullSet.npz','line_2_fullSet.npz','line_3_fullSet.npz','line_4_fullSet.npz','line_5_fullSet.npz','line_6_fullSet.npz','line_7_fullSet.npz','line_8_fullSet.npz','line_9_fullSet.npz' \
                 ,'line_10_fullSet.npz','line_11_fullSet.npz','line_12_fullSet.npz','line_13_fullSet.npz','line_14_fullSet.npz','line_15_fullSet.npz']
    
    inputFiles = []
    inputFiles.extend(julianFiles)
    inputFiles.extend(nikeFiles)
    inputFiles.extend(stephanFiles)
    inputFiles.extend(nadjaFiles)
    inputFiles.extend(lineFiles)
    
    inputFiles.sort()
    
    return inputFiles


def analyseBias():

    dsNames = getAllDataSetNames()
    dataSets = []
    for dsName in dsNames:
        dataSets.append(createDataSetFromFile(dsName))
    
    meanDrifts = []
    for gesture in range(0,10):
        signals = []
        for dataSet in dataSets:
            signals.extend( dataSet.getAllSignals(gesture, 2))
        signalSums = np.zeros((len(signals),9))
        for i, signal in enumerate(signals):
            signalSums[i,:] = np.sum(signal[:,0:9],0)
        meanDrifts.append( np.mean(signalSums,0)) 
    
    plt.figure()
    for meanDrift in meanDrifts:
        
        plt.plot(meanDrift)

    pass



def makeScatterPlots(totalTotalGesturePower,totalTotalGestureRotation,totalTotalGestureLenghts):
    
    gesturePowerMeans = map(np.mean,totalTotalGesturePower)
    gestureLengthMeans = map(np.mean,totalTotalGestureLenghts)
    gestureRotationMeans = map(np.mean,totalTotalGestureRotation)
    pdfScatterFilePath = resultsPath+'pdf/PowerLengthScatter.pdf'
    pp = PdfPages(pdfScatterFilePath)
    plt.figure()
    plt.title('Gesture power vs length')
    plt.xlabel('Absolute power in m/s^2')
    plt.ylabel('Absolute rotation in rad')
    plt.scatter(gesturePowerMeans, gestureLengthMeans, marker='o')
    
    for i in range(0,16):
        plt.annotate(str(i), xy=(gesturePowerMeans[i], gestureLengthMeans[i]), xytext=(gesturePowerMeans[i]+5, gestureLengthMeans[i]+5))
    pp.savefig()
    plt.figure()
    plt.title('Gesture power vs rotation')
    plt.scatter(gesturePowerMeans, gestureRotationMeans,marker='o')
    plt.xlabel('Power')
    plt.ylabel('Rotation')
    for i in range(0,16):
        plt.annotate(str(i), xy=(gesturePowerMeans[i], gestureRotationMeans[i]), xytext=(gesturePowerMeans[i], gestureRotationMeans[i]))
    pp.savefig()
    pp.close()
    
    fig = plt.figure(figsize=(10,10))
    plt.title('Gesture power vs rotation')

    plt.xlabel('Power')
    plt.ylabel('Rotation')
    cmap = mpl.cm.Set1
    legendEntries = []
    legendLabels = []
    for i in range(len(totalTotalGesturePower)):
        legEnt = plt.scatter(totalTotalGesturePower[i], totalTotalGestureRotation[i],marker='.',color=cmap(i / float(len(totalTotalGesturePower))) )
        legendEntries.append(legEnt)
        legendLabels.append(i)
    plt.legend(legendEntries,legendLabels)
    plt.scatter(gesturePowerMeans, gestureRotationMeans,marker='o')
    for i in range(0,16):
        plt.annotate(str(i), xy=(gesturePowerMeans[i], gestureRotationMeans[i]), xytext=(gesturePowerMeans[i], gestureRotationMeans[i]))
    
    pdfIndvScatterFilePath = resultsPath+'pdf/PowerLengthScatter_indv.pdf'
    pp = PdfPages(pdfIndvScatterFilePath)
    pp.savefig()
    
    plt.xlim((0,750))
    plt.ylim((0,300))
    
    pp.savefig()
    pp.close()




    
    
    
def main():
    pass
if __name__ == '__main__':    
    plt.close('all')
    resultsPath = getProjectPath()+'results/'
    
    now = datetime.datetime.now()
    resultsPath = getProjectPath()+'results/'
    pdfFileName = 'gestureAnalysis.pdf'
    pdfFilePath = resultsPath+'pdf/'+pdfFileName
    pp = PdfPages(pdfFilePath)
    print pdfFilePath
    
        
    #inputFiles = ['stephan_0_0.npz', 'stephan_0_1.npz','julian_0_0.npz','julian_0_1.npz','nike_0_0.npz','nike_0_1.npz']
    #secondInputFiles = ['stephan_1_0.npz','stephan_1_1.npz','julian_1_0.npz','julian_1_1.npz','nike_1_0.npz','nike_1_1.npz']
    #inputFiles = ['nadja_0_1.npz', 'nadja_0_2.npz', 'nadja_0_3.npz']
    #testFiles = ['lana_0_0.npz','lana_1_0.npz','stephan_0_2.npz','stephan_1_2.npz']
    
    dataSets=[]
    
    inputFiles = getAllDataSetNames()
    
    
    totalTotalGestureLenghts = []
    totalTotalGesturePower = []
    totalTotalGestureAvgPower=[]
    totalTotalGestureRotation = []
    for gestureNr in range(0,16):
        totalSignalLengths = []
        totalSignalPowers = []
        totalSignalAvgPowers = []
        totalSignalRotation = []
        totalFileNames = []
        for iFile in inputFiles:
            
            ds =createDataSetFromFile(iFile)
            dataSets.append(ds)
            signals = ds.getAllSignals(gestureNr, 2)
            nSignals = len(signals)
            if nSignals > 0:
                signalLengths = np.zeros((nSignals,1))
                signalPower = np.zeros((nSignals,1))
                signalRotation = np.zeros((nSignals,1))
                signalAvgPower = np.zeros((nSignals,1))
                
                for i in range(0,nSignals):
                    signal = signals[i]
                    
                    #signal lenght
                    rows, cols = signal.shape
                    signalLengths[i] = rows
                    
                    #signal power
                    power = np.sum(np.abs(signal[:,6:9]))
                    signalPower[i] = power
                    
                    signalAvgPower[i] = power/rows
                    
                    #signal power
                    rotation = np.sum(np.abs(signal[:,3:6]))
                    signalRotation[i] = rotation
                    
                totalSignalLengths.append(signalLengths)
                totalSignalPowers.append(signalPower)
                totalSignalRotation.append(signalRotation)
                totalSignalAvgPowers.append(signalAvgPower)
                totalFileNames.append(iFile)
                
        if len(totalFileNames) != 0:
            fig = plt.figure()
            plt.boxplot(totalSignalLengths, labels=totalFileNames)
            plt.ylim((0,80))
            plt.setp( fig.get_axes()[0].xaxis.get_majorticklabels(), rotation=70 )
            plt.title('length - Gesture '+str(gestureNr))
            plt.tight_layout()
            pp.savefig()
            
            fig = plt.figure()
            plt.boxplot(totalSignalPowers, labels=totalFileNames)
            plt.ylim((0,600))
            plt.setp( fig.get_axes()[0].xaxis.get_majorticklabels(), rotation=70 )
            plt.title('power - Gesture '+str(gestureNr))
            plt.tight_layout()
            pp.savefig()
            
            fig = plt.figure()
            plt.boxplot(totalSignalAvgPowers, labels=totalFileNames)
            plt.ylim((0,60))
            plt.setp( fig.get_axes()[0].xaxis.get_majorticklabels(), rotation=70 )
            plt.title('avg. power - Gesture '+str(gestureNr))
            plt.tight_layout()
            pp.savefig()
            
            fig = plt.figure()
            plt.boxplot(totalSignalRotation, labels=totalFileNames)
            plt.ylim((0,1000))
            plt.setp( fig.get_axes()[0].xaxis.get_majorticklabels(), rotation=70 )
            plt.title('rotation - Gesture '+str(gestureNr))
            plt.tight_layout()
            pp.savefig()
            totalTotalGestureLenghts.append(np.concatenate(tuple(totalSignalLengths)))
            totalTotalGesturePower.append(np.concatenate(tuple(totalSignalPowers)))
            totalTotalGestureAvgPower.append(np.concatenate(tuple(totalSignalAvgPowers)))
            totalTotalGestureRotation.append(np.concatenate(tuple(totalSignalRotation)))
        else:
            print iFile
        
    fig = plt.figure()
    plt.title('Length')
    plt.boxplot(totalTotalGestureLenghts)
    pp.savefig()
    plt.figure()
    plt.title('Power')
    plt.boxplot(totalTotalGesturePower)
    pp.savefig()
    plt.figure()
    plt.title('Avg. Power')
    plt.boxplot(totalTotalGestureAvgPower)
    pp.savefig()
    plt.figure()
    plt.title('Rotation')
    plt.boxplot(totalTotalGestureRotation)
    pp.savefig()
    
    pp.close()
    plt.close('all')
    
    
    # ---------------------------------------------------------------------------------- #
    # Making scatter plots:
    # ---------------------------------------------------------------------------------- #
    makeScatterPlots(totalTotalGesturePower,totalTotalGestureRotation,totalTotalGestureLenghts)
    
    
    #print 'std deviation: '+str(np.std(appendDS(dataSets, [0,1,2,3,4,5])[0],0))
    #print 'max: ' +str(np.max(np.abs(appendDS(dataSets, [0,1,2,3,4,5])[0]),0))