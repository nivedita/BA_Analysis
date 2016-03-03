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
    inputFileName = ("nadja_fitted_10.csv")
    pdfFileName = now.strftime("%Y-%m-%d-%H-%M")+'_'+inputFileName+'.pdf'
    pdfFilePath = resultsPath+pdfFileName
    
    
    pp = PdfPages(pdfFilePath)
    
        
    reservoir = Oger.nodes.ReservoirNode()
    readoutnode = Oger.nodes.RidgeRegressionNode()
    flow = mdp.Flow( [reservoir,readoutnode])

    
#    testData = inputData[350:,:]
#    testTargets = targets[350:,:]
#    inputData = inputData[:350,:]
#    targets = targets[:350,:]
#    inputData = multiplyData(inputData, 5)
#    targets = multiplyData(targets, 5)
#    testData = multiplyData(testData, 5) 
#    testTargets = multiplyData(testTargets, 5)
#    readOutTrainingData = numpy.atleast_2d(targets[:,2]).T



    #data = [[(inputData,readOutTrainingData),(inputData,readOutTrainingData),(inputData,readOutTrainingData)],[(inputData,readOutTrainingData),(inputData,readOutTrainingData),(inputData,readOutTrainingData)]]
    #data = [splitBySignals(inputData, targets, 2),splitBySignals(inputData, targets, 2)]
    a = DataSet.DataSet('nadja_fitted_0.csv')
    b = DataSet.DataSet('nadja_0_1.csv')
    c = DataSet.DataSet('nadja_0_2.csv')
    d = DataSet.DataSet('nadja_0_3.csv')
    
    data = [[a.getDataForTraining(useFused, useGyro, useAcc, 2,4),b.getDataForTraining(useFused, useGyro, useAcc, 2,4),d.getDataForTraining(useFused, useGyro, useAcc, 2,4)], \
            [a.getDataForTraining(useFused, useGyro, useAcc, 2,4),b.getDataForTraining(useFused, useGyro, useAcc, 2,4),d.getDataForTraining(useFused, useGyro, useAcc, 2,4)]]
    #flow.train(data)

#def startGridSearch():    
    gridsearch_parameters = {reservoir:{'spectral_radius':mdp.numx.arange(0.2, 0.8, 0.2),'output_dim':[4, 40, 400],'input_scaling': mdp.numx.arange(0.8, 1.4, 0.2),'_instance':range(5)},readoutnode:{'ridge_param':[0.00000001, 0.0001, 0.1]}}
    #gridsearch_parameters = {reservoir:{'spectral_radius':mdp.numx.arange(0.6, 1.2, 0.1),'input_scaling': mdp.numx.arange(0.8, 1.4, 0.1),'_instance':range(2)}}
    opt = Oger.evaluation.Optimizer(gridsearch_parameters, Oger.utils.nrmse)
    #opt.grid_search(data, flow, n_folds=3, cross_validate_function=Oger.evaluation.leave_one_out)
    opt.grid_search(data, flow, n_folds=3, cross_validate_function=Oger.evaluation.n_fold_random)
    
    #gridsearch_parameters = {reservoir:{'_instance':range(5), 'spectral_radius':mdp.numx.arange(0.8, 1.1, 0.1)}}
    #opt = Oger.evaluation.Optimizer(gridsearch_parameters, Oger.utils.nrmse)
    #opt.grid_search(data, flow, cross_validate_function=Oger.evaluation.n_fold_random, n_folds=2)
    plt.figure(3)
    plt.clf
    
    if gridsearch_parameters.has_key(readoutnode):
        opt.plot_results([(reservoir, '_instance'),(readoutnode, 'ridge_param'),(reservoir, 'output_dim')],plot_variance=False)
    else:
        opt.plot_results([(reservoir, '_instance')],plot_variance=False)
        
    pp.savefig()
    
    
    bestFlow = opt.get_optimal_flow(True)
    bestFlow.train(data)
    prediction = bestFlow([a.getDataForTraining(useFused, useGyro, useAcc, 2)[0]])
    
    
    plt.figure(2)
    plt.clf()
    plt.subplot()
    plt.title('Prediction on training')
    plt.plot(prediction)
    plt.subplot()
    plt.plot(numpy.atleast_2d(a.getDataForTraining(useFused, useGyro, useAcc, 2)[1]))
    plt.subplot()
    plt.plot(a.getDataForTraining(useFused, useGyro, useAcc, 2)[0]/numpy.max(a.getDataForTraining(useFused, useGyro, useAcc, 2)[0]))
    pp.savefig()
   
   
    t_prediction = bestFlow([c.getDataForTraining(useFused, useGyro, useAcc, 2)[0]])
    plt.figure()
    plt.clf()
    plt.subplot()
    plt.title('Prediction on test')
    plt.plot(t_prediction)
    plt.subplot()
    plt.plot(c.getDataForTraining(useFused, useGyro, useAcc, 2)[1])
    plt.subplot()
    #plt.plot(testData/numpy.max(inputData))
    pp.savefig()
      
   
   
    pp.close();  
    
       


    result = [str(now),inputFileName,str(opt.get_minimal_error())]
    result.extend(['fused',str(useFused),'gyro',str(useGyro),'acc',str(useAcc)])


    
    for a in opt.optimization_dict.get(reservoir).iterkeys():
        result.append(a)
        result.append(opt.optimization_dict.get(reservoir).get(a))
    if(opt.optimization_dict.get(readoutnode) != None):
        for a in opt.optimization_dict.get(readoutnode).iterkeys():
            result.append(a)
            result.append(opt.optimization_dict.get(readoutnode).get(a))

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



