import numpy
import matplotlib


import matplotlib.pyplot as plt
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
from DataSet import createDataSetFromFile

if __name__ == '__main__':
    
    plt.close('all')
    resultsPath = 'C:\Users\Steve\Documents\Eclipse Projects\BA_Analysis\\results\\'
    
    
        
    inputFiles = ['stephan_0_0.npz', 'stephan_0_1.npz','julian_0_0.npz','julian_0_1.npz','nike_0_0.npz','nike_0_1.npz']
    secondInputFiles = ['stephan_1_0.npz','stephan_1_1.npz','julian_1_0.npz','julian_1_1.npz','nike_1_0.npz','nike_1_1.npz']
    #inputFiles = ['nadja_0_1.npz', 'nadja_0_2.npz', 'nadja_0_3.npz']
    testFiles = ['lana_0_0.npz','lana_1_0.npz','stephan_0_2.npz','stephan_1_2.npz']
    julianFiles = ['julian_0_fullSet.npz','julian_1_fullSet.npz','julian_2_fullSet.npz','julian_3_fullSet.npz','julian_4_fullSet.npz','julian_5_fullSet.npz','julian_8_fullSet.npz','julian_9_fullSet.npz']
    inputFiles.extend(secondInputFiles)
    inputFiles.extend(testFiles)
    inputFiles.extend(julianFiles)
    inputFiles.sort()
    
    totalTotalGestureLenghts = []
    totalTotalGesturePower = []
    totalTotalGestureAvgPower=[]
    for gestureNr in range(0,10):
        totalSignalLengths = []
        totalSignalPowers = []
        totalSignalAvgPowers = []
        totalFileNames = []
        for iFile in inputFiles:
            
            ds =createDataSetFromFile(iFile)
            signals = ds.getAllSignals(gestureNr, 2)
            nSignals = len(signals)
            if nSignals > 0:
                signalLengths = np.zeros((nSignals,1))
                signalPower = np.zeros((nSignals,1))
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
                totalSignalLengths.append(signalLengths)
                totalSignalPowers.append(signalPower)
                totalSignalAvgPowers.append(signalAvgPower)
                totalFileNames.append(iFile)
                
        if len(totalFileNames) != 0:
            fig = plt.figure()
            plt.boxplot(totalSignalLengths, labels=totalFileNames)
            plt.ylim((0,80))
            plt.setp( fig.get_axes()[0].xaxis.get_majorticklabels(), rotation=70 )
            plt.title('length - Gesture '+str(gestureNr))
            plt.tight_layout()
            
            fig = plt.figure()
            plt.boxplot(totalSignalPowers, labels=totalFileNames)
            plt.ylim((0,600))
            plt.setp( fig.get_axes()[0].xaxis.get_majorticklabels(), rotation=70 )
            plt.title('power - Gesture '+str(gestureNr))
            plt.tight_layout()
            
            fig = plt.figure()
            plt.boxplot(totalSignalAvgPowers, labels=totalFileNames)
            plt.ylim((0,60))
            plt.setp( fig.get_axes()[0].xaxis.get_majorticklabels(), rotation=70 )
            plt.title('avg. power - Gesture '+str(gestureNr))
            plt.tight_layout()
            
            totalTotalGestureLenghts.append(np.concatenate(tuple(totalSignalLengths)))
            totalTotalGesturePower.append(np.concatenate(tuple(totalSignalPowers)))
            totalTotalGestureAvgPower.append(np.concatenate(tuple(totalSignalAvgPowers)))
            
        
    fig = plt.figure()
    plt.title('Length')
    plt.boxplot(totalTotalGestureLenghts)
    plt.figure()
    plt.title('Power')
    plt.boxplot(totalTotalGesturePower)
    plt.figure()
    plt.title('Avg. Power')
    plt.boxplot(totalTotalGestureAvgPower)
    
    