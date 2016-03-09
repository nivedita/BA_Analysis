'''
Created on 17.02.2016

@author: Steve
'''


import numpy
import matplotlib
import matplotlib.pyplot as plt
import scipy
import mdp
import csv
import Oger
import datetime
from matplotlib.backends.backend_pdf import PdfPages
from Server import *
import threading
import DataSet
from sklearn.metrics import f1_score

def transformToDelta(vals):
    newVals = numpy.zeros((len(vals),len(vals[0])))
    for i in range(1,len(vals)):
        newVals[i-1] = vals[i]-vals[i-1]
    return newVals

def readFileToNumpy(fileName):
    reader=csv.reader(open(fileName,"rb"),delimiter=';')
    x=list(reader)
    return numpy.array(x[1:]).astype('float')

def centerAndNormalize(inputData):
    means = numpy.mean(inputData, 0)
    centered = inputData - means
    vars = numpy.std(centered, 0)
    normalized = centered/vars
    return normalized

def multiplyData(data, multiplier):
    newData = data
    for i in range(0,multiplier):
        newData = numpy.append(newData, data, 0)
        newData = numpy.append(newData, numpy.zeros((50,len(data[0]))), 0)
    return newData

def separateInputData(fileData):
    fused = numpy.atleast_2d(fileData[:,0:3])
    gyro = numpy.atleast_2d(fileData[:,3:6])
    acc = numpy.atleast_2d(fileData[:,6:9])
    targets = numpy.atleast_2d(fileData[:,9:])
    return fused, gyro, acc, targets

def runningAverage(inputData, width):
    target = numpy.zeros((len(inputData),1))
    for i in range(width,len(inputData-width)):
            target[i] = numpy.mean(inputData[i-width:i+width])
    return target

def writeToReportFile(text):
    with open('C:\Users\Steve\Documents\Eclipse Projects\BA_Analysis\\results\\report.csv', 'ab') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(text)

def splitBySignals(inputData,target,targetCol=2):
    startInd = 0
    endInd = 0
    splitted = []
    for i in range(1,len(inputData)):
        endInd = i
        if (target[i-1,targetCol] > target[i,targetCol]):
            ins = inputData[startInd:endInd,:]
            outs = numpy.atleast_2d(target[startInd:endInd,targetCol]).T
            lens = len(ins)
            zeros = numpy.zeros((100-lens,len(ins[0])))
            ins = numpy.append(ins,zeros,0)
            outs = numpy.append(outs,numpy.zeros((100-lens,len(outs[0]))),0)                    
            tup = (ins,outs)
            splitted.append(tup)
            
            startInd = endInd
    return splitted

def calcWeightedAverage(input_signal, target_signal):
    global tresholdF1
    
    nDataPoints = len(input_signal.flatten())
    signals = []
    signalValue = 0
    nSignalPoints = 0
    for i in range(1,nDataPoints):
        if target_signal[i] == 1:
            signalValue = signalValue + input_signal[i]
            nSignalPoints = nSignalPoints+1
        elif target_signal[i] == 0 & target_signal[i-1] == 1:
            signals.append(signalValue/nSignalPoints)
            signalValue = 0
            nSignalPoints = 0
        else:
            pass
            
            
def calcSingleValueF1Score(input_signal, target_signal):
    nDataPoints = len(input_signal.flatten())
    bin_input_signal = numpy.ones((nDataPoints,1))
    bin_input_signal[input_signal < 0] = 0
    bin_target_signal = numpy.ones((nDataPoints,1))
    bin_target_signal[input_signal < 0] = 0
    score = f1_score(target_signal.astype('int'),bin_input_signal.astype('int'),average='binary')
    return 1-score

def calcF1Score(input_signal, target_signal):
    treshold = 0.5
    nDataPoints = len(input_signal.flatten())
    t_target = numpy.copy(target_signal)
    n_truePositive = 0
    n_falsePositive = 0
    i = 0
    while i < nDataPoints:
        n_true = 0
        n_false = 0
        removeTargetSignal = False
        while i < nDataPoints and input_signal[i] > treshold:
            if t_target[i] == 1:
                n_true = n_true + 1
            else: 
                n_false = n_false +1
            i = i+1
        if n_true > n_false:
            n_truePositive = n_truePositive + 1
            removeTargetSignal = True
        elif n_true < n_false:
            n_falsePositive = n_falsePositive + 1
        if removeTargetSignal:
            j = i
            while (j < nDataPoints) and (t_target[j] == 1) :   #remove this positive
                t_target[j] = 0
                j = j+1
        i = i+1
    
    
    n_totalPositives = 0
    lastVal = 0
    for i in range(0,nDataPoints):
        if target_signal[i] > lastVal:
            n_totalPositives = n_totalPositives+1
        lastVal = target_signal[i] 
    n_falseNegative = n_totalPositives - n_truePositive
    
    f1 = (2.*n_truePositive)/(2.*n_truePositive + n_falseNegative + n_falsePositive)
    print 1-f1   
    return 1-f1

def startServer():
    pass

    #address = ('192.168.0.115', 11111) # let the kernel give us a port
    #server = EchoServer(address, MyTCPHandler)
    #ip, port = server.server_address # find out what port we were given

    #t = threading.Thread(target=server.serve_forever)
    #t.setDaemon(True) # don't hang on exit
    #t.start()





if __name__ == '__main__':




    useFused = True
    useGyro = True
    useAcc = True
    
    plt.close('all')
    now = datetime.datetime.now()
    resultsPath = 'C:\Users\Steve\Documents\Eclipse Projects\BA_Analysis\\results\\'
    name = input('Name:')
    pdfFileName = now.strftime("%Y-%m-%d-%H-%M")+'_'+name+'.pdf'
    pdfFilePath = resultsPath+pdfFileName
    
    #inputFiles = ['stephan_0_0.npz', 'stephan_0_1.npz', 'stephan_0_2.npz','nadja_0_1.npz', 'nadja_0_2.npz', 'nadja_0_3.npz']
    inputFiles = ['nadja_0_1.npz', 'nadja_0_2.npz', 'nadja_0_3.npz']
    testFile = 'daniel_0_0.npz'
    
    pp = PdfPages(pdfFilePath)
    
        
    reservoir = Oger.nodes.ReservoirNode()
    readoutnode = Oger.nodes.RidgeRegressionNode()
    flow = mdp.Flow( [reservoir,readoutnode])

    
    trainSets = []
    testSets = []
    dataStep = []
    for iFile in inputFiles:
        set = DataSet.createDataSetFromFile(iFile)
        ds = DataSet.createDataSetFromFile('stephan_1_0.npz')
        ds.targets = numpy.ones(ds.acc.shape) * (-1)
        
        dataStep.append((numpy.append(set.getMinusPlusDataForTraining(0,useFused, useGyro, useAcc, 2)[0], \
                                     ds.getMinusPlusDataForTraining(0,useFused, useGyro, useAcc, 2)[0],0), \
                         numpy.append(set.getMinusPlusDataForTraining(0,useFused, useGyro, useAcc, 2)[1], \
                                     ds.getMinusPlusDataForTraining(0,useFused, useGyro, useAcc, 2)[1],0)))
    data = [dataStep,dataStep]
    #a = DataSet.createDataSetFromFile('nadja_fitted_0.csv') broken muss rekonstruiert werden
    #b = DataSet.createDataSetFromFile('nadja_0_1.npz')
    #c = DataSet.createDataSetFromFile('nadja_0_2.npz')
    #d = DataSet.createDataSetFromFile('nadja_0_3.npz')
    testSets.append(DataSet.createDataSetFromFile(testFile))
    testSets.append(DataSet.createDataSetFromFile('stephan_1_0.npz'))
    
    #data = [[b.getMinusPlusDataForTraining(useFused, useGyro, useAcc, 2),c.getMinusPlusDataForTraining(useFused, useGyro, useAcc, 2),d.getMinusPlusDataForTraining(useFused, useGyro, useAcc, 2)], \
    #        [b.getMinusPlusDataForTraining(useFused, useGyro, useAcc, 2),c.getMinusPlusDataForTraining(useFused, useGyro, useAcc, 2),d.getMinusPlusDataForTraining(useFused, useGyro, useAcc, 2)]]

    gridsearch_parameters = {reservoir:{'spectral_radius':mdp.numx.arange(0.6, 1.3, 0.1),'output_dim':[40,400],'input_scaling': mdp.numx.arange(1.5, 2.1, 0.1),'_instance':range(6)},readoutnode:{'ridge_param':[0.00000001,0.000001,0.0001]}}
    #gridsearch_parameters = {reservoir:{'spectral_radius':mdp.numx.arange(0.6, 1.2, 0.1),'input_scaling': mdp.numx.arange(0.8, 1.4, 0.1),'_instance':range(2)}}
    #opt = Oger.evaluation.Optimizer(gridsearch_parameters, calcF1Score)
    opt = Oger.evaluation.Optimizer(gridsearch_parameters, Oger.utils.nrmse)
    opt.grid_search(data, flow, n_folds=3, cross_validate_function=Oger.evaluation.n_fold_random)
    

    

    
    if gridsearch_parameters.has_key(readoutnode):
        plt.figure()
        opt.plot_results([(reservoir, '_instance'),(readoutnode, 'ridge_param'),(reservoir, 'output_dim')],plot_variance=False)
        pp.savefig()
        plt.figure()
        opt.plot_results([(reservoir, '_instance'),(reservoir, 'spectral_radius'),(reservoir, 'input_scaling')],plot_variance=False)
        pp.savefig()
    else:
        opt.plot_results([(reservoir, '_instance')],plot_variance=False)
        
    
    
    bestFlow = opt.get_optimal_flow(True)
    bestFlow.train(data)
    
    

    
    nInputFiles = len(inputFiles)
    fig, axes = plt.subplots(nInputFiles, 1)
    plt.title('Prediction on training')  
    i = 0 
    for row in axes:
        prediction = bestFlow([data[0][i][0]])
        row.set_title(inputFiles[i])
        row.plot(prediction)
        row.plot(numpy.atleast_2d(data[0][i][1]))
        i = i+1
    #plt.plot(data[0][0][0])
    pp.savefig()
   
   
    for set in testSets:
        t_prediction = bestFlow([set.getMinusPlusDataForTraining(0,useFused, useGyro, useAcc, 2)[0]])
        plt.figure()
        plt.clf()
        plt.subplot(211)
        plt.title('Prediction on test')
        plt.plot(t_prediction)
        plt.plot(set.getMinusPlusDataForTraining(0,useFused, useGyro, useAcc, 2)[1])
        plt.subplot(212)
        plt.title('Smoothed prediction')
        plt.plot(runningAverage(t_prediction, 10))
        plt.plot(set.getMinusPlusDataForTraining(0,useFused, useGyro, useAcc, 2)[1])
        print calcF1Score(t_prediction, set.getMinusPlusDataForTraining(0,useFused, useGyro, useAcc, 2)[1])
        pp.savefig()
   
   
    pp.close();  
    
    
       


    result = [str(now),name,inputFiles,testFile,opt.loss_function,str(opt.get_minimal_error()[0])]
    result.extend(['fused',str(useFused),'gyro',str(useGyro),'acc',str(useAcc)])
        
        
    for a in opt.get_minimal_error()[1].iterkeys():
        result.append(a)
        for attribute in opt.get_minimal_error()[1].get(a).iterkeys():
            result.append(attribute)
            result.append('['+str(opt.get_minimal_error()[1].get(a).get(attribute))+']')
            
            




    result.append('gridSpace:')
    for a in opt.optimization_dict.get(reservoir).iterkeys():
        result.append(a)
        result.append(opt.optimization_dict.get(reservoir).get(a))
    if(opt.optimization_dict.get(readoutnode) != None):
        for a in opt.optimization_dict.get(readoutnode).iterkeys():
            result.append(a)
            result.append(str(opt.optimization_dict.get(readoutnode).get(a)))

    result.extend(['paraList',opt.parameter_ranges])
    result.extend(['errors',numpy.array2string(opt.errors).replace('\n', ',').replace('  ',',').replace(',,',',').replace(',,',',')])
    


    
    result.append('=HYPERLINK(\"'+pdfFilePath+'\")')
    
    writeToReportFile(result)
    
    #gridsearch_parameters = {reservoir:{'_instance':range(5), 'spectral_radius':mdp.numx.arange(0.6, 1.3, 0.1)}}
    #opt1D = Oger.evaluation.Optimizer(gridsearch_parameters, Oger.utils.nrmse)
    #opt1D.grid_search(data, flow, n_folds=3, cross_validate_function=Oger.evaluation.n_fold_random)
    #opt1D.plot_results([(reservoir, '_instance')])
    
    #gridsearch_parameters = {reservoir:{'spectral_radius':mdp.numx.arange(0.6, 1.3, 0.2),'input_scaling': mdp.numx.arange(0.5, .8, 0.1),'_instance':range(5)}}
    #opt2D = Oger.evaluation.Optimizer(gridsearch_parameters, Oger.utils.nrmse)
    #errors = opt2D.grid_search(data, flow, n_folds=3, cross_validate_function=Oger.evaluation.n_fold_random)
    #opt2D.plot_results([(reservoir, '_instance')])



