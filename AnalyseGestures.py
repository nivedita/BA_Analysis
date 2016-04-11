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
from mpl_toolkits.mplot3d import Axes3D
    


def getAllDataSetNames():
    julianFiles = ['julian_0_fullSet.npz','julian_1_fullSet.npz','julian_2_fullSet.npz','julian_3_fullSet.npz','julian_4_fullSet.npz','julian_5_fullSet.npz','julian_6_fullSet.npz','julian_7_fullSet.npz','julian_8_fullSet.npz','julian_9_fullSet.npz', \
                   'julian_10_fullSet.npz','julian_11_fullSet.npz','julian_12_fullSet.npz','julian_13_fullSet.npz','julian_14_fullSet.npz','julian_15_fullSet.npz']
    nikeFiles = ['nike_0_fullSet.npz','nike_1_fullSet.npz','nike_2_fullSet.npz','nike_3_fullSet.npz','nike_4_fullSet.npz','nike_5_fullSet.npz','nike_6_fullSet.npz','nike_7_fullSet.npz','nike_8_fullSet.npz','nike_9_fullSet.npz', \
                 'nike_10_fullSet.npz','nike_11_fullSet.npz','nike_12_fullSet.npz','nike_13_fullSet.npz','nike_14_fullSet.npz','nike_15_fullSet.npz']
    stephanFiles = ['stephan_0_fullSet.npz','stephan_1_fullSet.npz','stephan_2_fullSet.npz','stephan_3_fullSet.npz','stephan_4_fullSet.npz','stephan_5_fullSet.npz','stephan_6_fullSet.npz','stephan_7_fullSet.npz','stephan_8_fullSet.npz','stephan_9_fullSet.npz']
    nadjaFiles = ['nadja_0_fullSet.npz','nadja_1_fullSet.npz','nadja_2_fullSet.npz','nadja_3_fullSet.npz','nadja_4_fullSet.npz','nadja_5_fullSet.npz','nadja_6_fullSet.npz','nadja_7_fullSet.npz','nadja_8_fullSet.npz','nadja_9_fullSet.npz']
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



def makeScatterPlots(totalTotalGesturePower,totalTotalGestureRotation,totalTotalGestureLenghts, xlim = None, ylim = None):
    
    gesturePowerMeans = map(np.mean,totalTotalGesturePower)
    gestureLengthMeans = map(np.mean,totalTotalGestureLenghts)
    gestureRotationMeans = map(np.mean,totalTotalGestureRotation)
    pdfScatterFilePath = resultsPath+'pdf/PowerLengthScatter'+title+'.pdf'
    pp = PdfPages(pdfScatterFilePath)
    plt.figure()
    plt.title('Gesture power vs length')
    plt.xlabel('Absolute power in m/s^2')
    plt.ylabel('Length in timesteps')
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
        legEnt = plt.scatter(totalTotalGesturePower[i], totalTotalGestureRotation[i],marker='o',color=cmap(i / float(len(totalTotalGesturePower))) )
        legendEntries.append(legEnt)
        legendLabels.append(i)
        plt.scatter(gesturePowerMeans[i], gestureRotationMeans[i],marker='o',s=100,color=cmap(i / float(len(totalTotalGesturePower))) )
    plt.legend(legendEntries,legendLabels)
    for i in range(0,16):
        plt.annotate(str(i), xy=(gesturePowerMeans[i], gestureRotationMeans[i]), xytext=(gesturePowerMeans[i], gestureRotationMeans[i]))
    
    pdfIndvScatterFilePath = resultsPath+'pdf/PowerLengthScatter'+title+'_indv.pdf'
    pp = PdfPages(pdfIndvScatterFilePath)
    pp.savefig()
    
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    
    pp.savefig()
    pp.close()


def makeScatterPlotIndv(totalTotalGesturePower,totalTotalGestureRotation,totalTotalGestureLenghts, title, xlim = None, ylim = None):
    
    cmaps = [mpl.cm.Reds_r,mpl.cm.Blues_r,mpl.cm.Greens_r,mpl.cm.Purples,mpl.cm.Oranges_r]
    classRanges = [4,2,2,2,6]
    gesturePowerMeans = map(np.mean,totalTotalGesturePower)
    gestureLengthMeans = map(np.mean,totalTotalGestureLenghts)
    gestureRotationMeans = map(np.mean,totalTotalGestureRotation)
    pdfScatterFilePath = resultsPath+'pdf/PowerLengthScatter'+title+'.pdf'
    pp = PdfPages(pdfScatterFilePath)
    plt.figure()
    plt.title('Gesture power vs length')
    plt.xlabel('Absolute power in m/s^2')
    plt.ylabel('Length in timesteps')
    for classNr in range(0,4):
        cmap = cmaps[classNr]
        classLen = classRanges[classNr]
        startOfClass = np.sum(classRanges[0:classNr],None, 'int')
        for i in range(startOfClass,startOfClass+classLen):
            plt.scatter(gesturePowerMeans[i], gestureLengthMeans[i],marker='o',s=100,color=cmap(i / float(len(totalTotalGesturePower))) )    
    for i in range(0,10):
        plt.annotate(str(i), xy=(gesturePowerMeans[i], gestureLengthMeans[i]), xytext=(gesturePowerMeans[i], gestureLengthMeans[i]))
    pp.savefig()
    plt.figure()
    plt.title('Gesture power vs rotation')
    for classNr in range(0,4):
        cmap = cmaps[classNr]
        classLen = classRanges[classNr]
        startOfClass = np.sum(classRanges[0:classNr],None, 'int')
        for i in range(startOfClass,startOfClass+classLen):
            plt.scatter(gesturePowerMeans[i], gestureRotationMeans[i],marker='o',s=100,color=cmap(i / float(len(totalTotalGesturePower))) )    
    plt.xlabel('Power')
    plt.ylabel('Rotation')
    for i in range(0,10):
        plt.annotate(str(i), xy=(gesturePowerMeans[i], gestureRotationMeans[i]), xytext=(gesturePowerMeans[i], gestureRotationMeans[i]))
    pp.savefig()
    pp.close()
    
    fig = plt.figure(figsize=(10,10))
    plt.title('Gesture power vs rotation')

    plt.xlabel('Power')
    plt.ylabel('Rotation')

    legendEntries = []
    legendLabels = []
    for classNr in range(0,4):
        cmap = cmaps[classNr]
        classLen = classRanges[classNr]
        startOfClass = np.sum(classRanges[0:classNr],None, 'int')
        for i in range(startOfClass,startOfClass+classLen):
            legEnt = plt.scatter(totalTotalGesturePower[i], totalTotalGestureRotation[i],marker='o',color=cmap(i / float(len(totalTotalGesturePower))) )
            legendEntries.append(legEnt)
            legendLabels.append(i)
            plt.scatter(gesturePowerMeans[i], gestureRotationMeans[i],marker='o',s=200,color=cmap(i / float(len(totalTotalGesturePower))) )
    plt.legend(legendEntries,totalGestureNames)
    
    for i in range(0,10):
        plt.annotate(str(i), xy=(gesturePowerMeans[i], gestureRotationMeans[i]), xytext=(gesturePowerMeans[i], gestureRotationMeans[i]))
    
    pdfIndvScatterFilePath = resultsPath+'pdf/PowerLengthScatter'+title+'_indv.pdf'
    pp = PdfPages(pdfIndvScatterFilePath)
    pp.savefig()
    
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    
    pp.savefig()
    pp.close()



def plot3dFused(signals,name):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cmap = mpl.cm.Set1
    for k, sig in enumerate(signals):
        for j, fused1 in enumerate(sig.getAllSignals()):
            fused1  = fused1[:,6:9]
            fusedInt1 = np.zeros(fused1.shape)
            for i in range(0,fused1.shape[0]):
                fusedInt1[i] = np.sum(fused1[:i],0)
            if j == 0:
                ax.plot(fusedInt1[:,0],fusedInt1[:,1],fusedInt1[:,2],c=cmap(k / float(len(signals))),label=str(k))
            else:
                ax.plot(fusedInt1[:,0],fusedInt1[:,1],fusedInt1[:,2],c=cmap(k / float(len(signals))))
            
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(name)
    ax.legend()

def plotGesturesVs():
    pass
#if __name__ == '__main__':
    ds1 = createDataSetFromFile('nadja_7_fullSet.npz')
    ds1signals = ds1.getAllSignals(7,2)
    ds2 = createDataSetFromFile('julian_7_fullSet.npz')
    ds2signals = ds2.getAllSignals(7,2)
    ds3 = createDataSetFromFile('julian_2_fullSet.npz')
    ds4 = createDataSetFromFile('julian_3_fullSet.npz')
    
    plt.switch_backend('Qt4Agg')
    
    
    axes, subs = plt.subplots(3, 1, True)
    print zip(map(len,ds1signals),map(len,ds2signals))
    for i in range(0,3):
        label=['Gest. 1 X','Gest. 1 Y','Gest. 1 Z']
        cmap = mpl.cm.Reds_r
        subs[0].plot(ds1signals[1][:,i] ,label=label[i] ,color=cmap(i / 3.0))
    for i in range(0,3):
        label=['Gest. 2 X','Gest. 2 Y','Gest. 2 Z']
        cmap = mpl.cm.Blues
        subs[0].plot(ds2signals[8][:,i] ,label=label[i] ,color=cmap((i+0.5) / 3.0))
    subs[0].set_title('Fused')
    
    for i in range(3,6):
        label=['Gest. 1 X','Gest. 1 Y','Gest. 1 Z']
        cmap = mpl.cm.Reds_r
        subs[1].plot(ds1signals[1][:,i] ,label=label[i-3] ,color=cmap((i-3) / 3.0))
    for i in range(3,6):
        label=['Gest. 2 X','Gest. 2 Y','Gest. 2 Z']
        cmap = mpl.cm.Blues
        subs[1].plot(ds2signals[8][:,i] ,label=label[i-3] ,color=cmap((i+1-3) / 3.0))
    subs[1].set_title('Rotation')
    
    for i in range(6,9):
        label=['Gest. 1 X','Gest. 1 Y','Gest. 1 Z']
        cmap = mpl.cm.Reds_r
        subs[2].plot(ds1signals[1][:,i] ,label=label[i-6] ,color=cmap((i-6) / 3.0))
    for i in range(6,9):
        label=['Gest. 2 X','Gest. 2 Y','Gest. 2 Z']
        cmap = mpl.cm.Blues 
        subs[2].plot(ds2signals[8][:,i] ,label=label[i-6] ,color=cmap((i+1-6) / 3.0))
    subs[2].set_title('Acceleration')
            
        
    subs[0].legend()
    subs[1].legend()
    subs[2].legend()


    for name in ['stephan','julian','nadja','line','nike']:
        ds1 = createDataSetFromFile(name+'_7_fullSet.npz')
        ds2 = createDataSetFromFile(name+'_7_fullSet.npz')
        ds3 = createDataSetFromFile(name+'_8_fullSet.npz')
        ds4 = createDataSetFromFile(name+'_8_fullSet.npz')
        plot3dFused([ds1, ds2,ds3,ds4],name)
        
    for j, signal in enumerate(createDataSetFromFile('nike_3_fullSet.npz').getAllSignals()):
        fig,axes = plt.subplots(2, 1, True)
        fused1  = signal[:,0:3]
        fusedInt1 = np.zeros(fused1.shape)
        fusedInt2 = np.zeros(fused1.shape)
        for i in range(1,fused1.shape[0]):
            fusedInt1[i] = np.sum(fused1[:i],0)
            fusedInt2[i] = fusedInt2[i-1]*0.95+fused1[i]
        axes[0].plot(fusedInt1)
        axes[1].plot(fusedInt2)
        plt.title(str(j))
  
def analyseGesture(gestNr):
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=3)
    results = []
    dsNames = ['line','nadja','stephan','nike','julian']
    dsNames.sort()
    
    for name in dsNames:
        results.append(analyseDataSet(name+'_'+str(gestNr)+'_fullSet.npz'))
    
    print 'DataSet' +'&' + 'Length'+ '&' + 'Power'+ '&' + 'Rotation'+ '&' + 'Fused'+ ' \\\\ \hline'
    
    #for i, result in enumerate(results):
    #    print dsNames[i] +'&' + str(result[0]) + '&' + str(result[1]) + '&' + str(result[2]) + '&' + str(result[3]) + ' \\\\ \hline'
    print " \\\\ \hline \n".join([" & ".join(map(str,line)) for line in results])    
def analyseDataSet(dataSetName):
    ds = createDataSetFromFile(dataSetName)
    signals = ds.getAllSignals()
    lengths = map(len,signals)
    power = map(normPower,signals)
    rot = map(normRot,signals)
    fused = map(normFused,signals)
    power_means =  map(np.mean,power)
    rot_means   =  map(np.mean,rot)
    fused_means =  map(np.mean,fused)
    print np.var(lengths)
    print np.var(power_means)
    print np.var(rot_means)
    print np.var(fused_means)
    return [np.var(lengths), np.var(power_means),np.var(rot_means),np.var(fused_means)]
    
   
def normPower(X):
    print X.shape
    return np.linalg.norm(X[:,6:9], None, 1) 
def normRot(X):
    return np.linalg.norm(X[:,3:6], None, 1) 
def normFused(X):
    return np.linalg.norm(X[:,0:3], None, 1) 


def plotDSAgainst(nr):
    plt.close('all')
    nadja = createDataSetFromFile('nadja_'+str(nr)+'_fullSet.npz')
    line = createDataSetFromFile('line_'+str(nr)+'_fullSet.npz')
    stephan = createDataSetFromFile('stephan_'+str(nr)+'_fullSet.npz')
    nike = createDataSetFromFile('nike_'+str(nr)+'_fullSet.npz')
    julian = createDataSetFromFile('julian_'+str(nr)+'_fullSet.npz')
    nadja.plot(2,False)
    line.plot(2,False) #falsch
    stephan.plot(2,False)
    nike.plot(2,False)
    julian.plot(2,False) #falsch
    

#def main():
#    pass
if __name__ == '__main__':    
    plt.close('all')
    resultsPath = getProjectPath()+'results/'
    
    now = datetime.datetime.now()
    resultsPath = getProjectPath()+'results/'
    pdfFileName = 'gestureAnalysis.pdf'
    pdfFilePath = resultsPath+'pdf/'+pdfFileName
    pp = PdfPages(pdfFilePath)
    print pdfFilePath
    
    totalGestureNames = ['snap left','snap right','snap forward','snap backward','bounce up','bounce down','turn left','turn right','shake l-r','shake u-d']
                         #'tap left','tap right','tap front','tap back','tap top','tap bottom']
    
        
    plt.switch_backend('Agg')
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
    totalTotalGestureAvgRotation = []
    lengthVariances = []
    
    for gestureNr in range(0,10):
        totalSignalLengths = []
        totalSignalPowers = []
        totalSignalAvgPowers = []
        totalSignalRotation = []
        totalSignalAvgRotation = []
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
                signalAvgRotation = np.zeros((nSignals,1))
                
                for i in range(0,nSignals):
                    signal = signals[i]
                    
                    #signal lenght
                    rows, cols = signal.shape
                    signalLengths[i] = rows
                    
                    #signal power
                    power = np.sum(np.linalg.norm(signal[:,6:9],2,1))
                    signalPower[i] = power
                    
                    signalAvgPower[i] = power/rows
                    
                    #signal power
                    rotation = np.sum(np.linalg.norm(signal[:,3:6],2,1))
                    signalRotation[i] = rotation

                    
                    signalAvgRotation[i] = rotation/rows
                totalSignalLengths.append(signalLengths)
                totalSignalPowers.append(signalPower)
                totalSignalRotation.append(signalRotation)
                totalSignalAvgPowers.append(signalAvgPower)
                totalSignalAvgRotation.append(signalAvgRotation)
                
                totalFileNames.append(iFile)

        if len(totalFileNames) != 0:
            fig = plt.figure()
            plt.boxplot(totalSignalLengths, labels=totalFileNames)
            plt.ylim((0,80))
            plt.setp( fig.get_axes()[0].xaxis.get_majorticklabels(), rotation=70 )
            plt.title('length - ' + totalGestureNames[gestureNr])
            plt.tight_layout()
            pp.savefig()
            
            fig = plt.figure()
            plt.boxplot(totalSignalPowers, labels=totalFileNames)
            plt.ylim((0,600))
            plt.setp( fig.get_axes()[0].xaxis.get_majorticklabels(), rotation=70 )
            plt.title('power - ' + totalGestureNames[gestureNr])
            plt.tight_layout()
            pp.savefig()
            
            fig = plt.figure()
            plt.boxplot(totalSignalAvgPowers, labels=totalFileNames)
            plt.ylim((0,60))
            plt.setp( fig.get_axes()[0].xaxis.get_majorticklabels(), rotation=70 )
            plt.title('avg. power - ' + totalGestureNames[gestureNr])
            plt.tight_layout()
            pp.savefig()
            
            fig = plt.figure()
            plt.boxplot(totalSignalRotation, labels=totalFileNames)
            plt.ylim((0,1000))
            plt.setp( fig.get_axes()[0].xaxis.get_majorticklabels(), rotation=70 )
            plt.title('rotation - ' + totalGestureNames[gestureNr])
            plt.tight_layout()
            pp.savefig()
            
            fig = plt.figure()
            plt.boxplot(totalSignalAvgRotation, labels=totalFileNames)
            plt.ylim((0,20))
            plt.setp( fig.get_axes()[0].xaxis.get_majorticklabels(), rotation=70 )
            plt.title('avg. rotation - ' + totalGestureNames[gestureNr])
            plt.tight_layout()
            pp.savefig()
            
            totalTotalGestureLenghts.append(np.concatenate(tuple(totalSignalLengths)))
            totalTotalGesturePower.append(np.concatenate(tuple(totalSignalPowers)))
            totalTotalGestureAvgPower.append(np.concatenate(tuple(totalSignalAvgPowers)))
            totalTotalGestureRotation.append(np.concatenate(tuple(totalSignalRotation)))
            totalTotalGestureAvgRotation.append(np.concatenate(tuple(totalSignalAvgRotation)))
        else:
            print iFile
    pp.close()    
    
    pdfFilePath = resultsPath+'pdf/totalGesture.pdf'
    pp = PdfPages(pdfFilePath)
    fig = plt.figure()
    plt.title('Length')
    plt.boxplot(totalTotalGestureLenghts, labels=totalGestureNames)
    plt.xticks(rotation=90)
    plt.tight_layout()
    pp.savefig()
    plt.figure()
    plt.title('Power')
    plt.boxplot(totalTotalGesturePower, labels=totalGestureNames)
    plt.xticks(rotation=90)
    plt.tight_layout()
    pp.savefig()
    plt.figure()
    plt.title('Avg. Power')
    plt.boxplot(totalTotalGestureAvgPower, labels=totalGestureNames)
    plt.xticks(rotation=90)
    plt.tight_layout()
    pp.savefig()
    plt.figure()
    plt.title('Rotation')
    plt.boxplot(totalTotalGestureRotation, labels=totalGestureNames)
    plt.xticks(rotation=90)
    plt.tight_layout()
    pp.savefig()
    plt.figure()
    plt.title('Avg. Rotation')
    plt.boxplot(totalTotalGestureAvgRotation, labels=totalGestureNames)
    plt.xticks(rotation=90)
    plt.tight_layout()
    pp.savefig()    
    pp.close()
    plt.close('all')
    
    
    # ---------------------------------------------------------------------------------- #
    # Making scatter plots:
    # ---------------------------------------------------------------------------------- #
    plt.switch_backend('Qt4Agg')
    
    makeScatterPlotIndv(totalTotalGesturePower,totalTotalGestureRotation,totalTotalGestureLenghts,'', (0,750), (0,300) )
    makeScatterPlotIndv(totalTotalGestureAvgPower,totalTotalGestureAvgRotation,totalTotalGestureLenghts,'_avg_', (0,30),(0,10))
    
    
    
    #print 'std deviation: '+str(np.std(appendDS(dataSets, [0,1,2,3,4,5])[0],0))
    #print 'max: ' +str(np.max(np.abs(appendDS(dataSets, [0,1,2,3,4,5])[0]),0))