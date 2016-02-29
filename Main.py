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


def transformToDelta(vals):
    newVals = numpy.zeros((len(vals),len(vals[0])))
    for i in range(1,len(vals)):
        newVals[i-1] = vals[i]-vals[i-1]
    return newVals

def readFileToNumpy(fileName):
    reader=csv.reader(open(fileName,"rb"),delimiter=',')
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
        newData = numpy.append(newData, numpy.zeros((numpy.random.randint(1000),31)), 0)
    return newData

def separateInputData(fileData):
    fused = numpy.atleast_2d(fileData[:,1:4])
    gyro = numpy.atleast_2d(fileData[:,4:7])
    acc = numpy.atleast_2d(fileData[:,7:10])
    targets = numpy.atleast_2d(fileData[:,10:])
    return fused, gyro, acc, targets

def writeToReportFile(text):
    with open('C:\Users\Steve\Documents\Eclipse Projects\BA_Analysis\\results\\report.csv', 'ab') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(text)


def startServer():
    pass

    #address = ('192.168.0.115', 11111) # let the kernel give us a port
    #server = EchoServer(address, MyTCPHandler)
    #ip, port = server.server_address # find out what port we were given

    #t = threading.Thread(target=server.serve_forever)
    #t.setDaemon(True) # don't hang on exit
    #t.start()





if __name__ == '__main__':




    useDelta = True
    useCenterAndNormalize = True
    useFused = True
    useGyro = True
    useAcc = True

    plt.close('all')
    now = datetime.datetime.now()
    resultsPath = 'C:\Users\Steve\Documents\Eclipse Projects\BA_Analysis\\results\\'
    inputFileName = '2016-02-27-17-36-01-nadja2.csv'
    pdfFileName = now.strftime("%Y-%m-%d-%H-%M")+'_'+inputFileName+'.pdf'
    pdfFilePath = resultsPath+pdfFileName
    
    
    pp = PdfPages(pdfFilePath)
    

    fileData = readFileToNumpy('C:\Users\Steve\Documents\Eclipse Projects\BA_Analysis\\'+inputFileName)
    fileData = multiplyData(fileData, 4)
    fused, gyro, acc, targets = separateInputData(fileData)
    
    if useDelta:
        fused = transformToDelta(fused)
        gyro  = transformToDelta(gyro)
        acc   = transformToDelta(acc)

    if useCenterAndNormalize:
        fused = centerAndNormalize(fused)
        gyro = centerAndNormalize(gyro)
        acc = centerAndNormalize(acc)

    plt.figure(1)
    plt.clf()
    plt.subplot(311)
    plt.title('Fused')
    plt.plot(fused)
    plt.plot(targets)
    
    plt.subplot(312)
    plt.title('Gyro')
    plt.plot(gyro)
    
    plt.subplot(313)
    plt.title('Acc')
    plt.plot(acc)
    pp.savefig()
    
    
    

    #formatedInputData = [numpy.expand_dims(result[0:1000,0],1).T]
    #formatedInputData[0] = formatedInputData[0].T
    #for i in range(1,8) :
    #    formatedInputData.append(numpy.expand_dims(result[0:1000,i],1).T)
    #    formatedInputData[i] = formatedInputData[i].T
    #    print(i)
    #target = result[0:1000,9]
    #target=[numpy.expand_dims(target, 1)]
    

    
    
    readOutTrainingData = numpy.atleast_2d(targets[:,0]).T
    data = [[None]]
    
    #data = [x[0:-1], zip(x[0:-1],y[0:-1])]
    reservoir = Oger.nodes.ReservoirNode(output_dim=400)
    readoutnode = Oger.nodes.RidgeRegressionNode()
    flow = mdp.Flow( [reservoir,readoutnode])


    inputData = numpy.empty((0,0))
    if useFused:
        inputData = fused
        
    if useGyro:
        if len(inputData) == 0:
            inputData = gyro
        else:
            inputData = numpy.append(inputData, gyro, 1)
            
    if useAcc:
        if len(inputData) == 0:
            inputData = acc
        else:
            inputData = numpy.append(inputData, acc, 1)
            
            
    data = [[(inputData,readOutTrainingData),(inputData,readOutTrainingData),(inputData,readOutTrainingData)],[(inputData,readOutTrainingData),(inputData,readOutTrainingData),(inputData,readOutTrainingData)]]
    #flow.train(data)
    

    
    
    gridsearch_parameters = {reservoir:{'spectral_radius':mdp.numx.arange(0.5, 1.11, 0.2),'input_scaling': mdp.numx.arange(0.8, 1.4, 0.2),'_instance':range(1)},readoutnode:{'ridge_param':[ 0.001, 1]}}
    opt = Oger.evaluation.Optimizer(gridsearch_parameters, Oger.utils.nrmse)
    errors = opt.grid_search(data, flow, n_folds=3, cross_validate_function=Oger.evaluation.n_fold_random)
    
    #gridsearch_parameters = {reservoir:{'_instance':range(5), 'spectral_radius':mdp.numx.arange(0.8, 1.1, 0.1)}}
    #opt = Oger.evaluation.Optimizer(gridsearch_parameters, Oger.utils.nrmse)
    #opt.grid_search(data, flow, cross_validate_function=Oger.evaluation.n_fold_random, n_folds=2)
    plt.figure(3)
    plt.clf
    opt.plot_results([(reservoir, '_instance'),(readoutnode, 'ridge_param')])
    pp.savefig()
    
    
    bestFlow = opt.get_optimal_flow(True)
    bestFlow.train(data)
    prediction = bestFlow([inputData])
    
    
    plt.figure(2)
    plt.clf()
    plt.subplot()
    plt.title('Prediction')
    plt.plot(prediction)
    plt.subplot()
    plt.plot(numpy.expand_dims(targets[:,0],1))
    plt.subplot()
    plt.plot(inputData/numpy.max(inputData))
    pp.savefig()
   
    pp.close();     


    result = [str(now),inputFileName,str(opt.get_minimal_error())]
    result.extend(['fused',str(useFused),'gyro',str(useGyro),'acc',str(useAcc)])
    result.extend(['delta',str(useDelta),'centerAndNormalize',str(useCenterAndNormalize)])

    
    for a in opt.optimization_dict.get(reservoir).iterkeys():
        result.append(a)
        result.append(opt.optimization_dict.get(reservoir).get(a))
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



