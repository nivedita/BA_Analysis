'''
Created on 17.02.2016

@author: Steve
'''

import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import mdp
import csv
import Oger
import datetime
import DataSet
import Evaluation
import OptDicts
import EvaluateTestFile

from Utils import getProjectPath
import sklearn
import sklearn.metrics
from sklearn.metrics import f1_score
from DataAnalysis import subPlot
from SparseNode import SparseNode










def runningAverage(inputData, width):
    inputData = np.atleast_2d(inputData)
    target = np.zeros((inputData.shape))
    for i in range(width,len(inputData-width)):
            target[i,:] = np.mean(inputData[i-width:i+width,:],0)
    return target

def writeToReportFile(text):
    print getProjectPath()+'results/report.csv'
    with open(getProjectPath()+'results/report.csv', 'ab') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(text)

def splitBySignals(dataStep):
#    pass
#if __name__ == '__main__':
    segments= []
    for input, target in dataStep:
        targetInt = np.argmax(Evaluation.addNoGestureSignal(target), 1)
        inds= np.where(targetInt[:-1]!= targetInt[1:])[0]
        lastInd = -1
        for ind in inds:
            if targetInt[ind] != np.max(targetInt):
                iSegment = input[lastInd+1:ind+1]
                tSegement = target[lastInd+1:ind+1]
                tSegement[0,:]=0
                tSegement[-1,:]=0
                segments.append((iSegment,tSegement))
                lastInd = ind
        ind = len(targetInt)-1
        iSegment = input[lastInd+1:ind+1]
        tSegement = target[lastInd+1:ind+1]
        tSegement[0,:]=0
        tSegement[-1,:]=0
        segments.append((iSegment,tSegement))
    return segments

def shuffleDataStep(dataStep, nFolds):
    segs = splitBySignals(dataStep)
    random.shuffle(segs)
    segs = [ segs[i::nFolds] for i in xrange(nFolds) ]
    dataStep=[]
    for segList in segs:
        ind = np.concatenate([x[0] for x in segList],0)
        t   = np.concatenate([x[1] for x in segList],0)
        dataStep.append((ind,t))
    return dataStep


            
 






if __name__ == '__main__':
    
    #Set up plt
    matplotlib.rcParams.update({'font.size': 20})    
    plt.switch_backend('Qt4Agg')
    plt.close('all')
    
    
    #===========================================================================
    # Give this run a name. 
    # If name equals 'test', no log will be generated
    #===========================================================================
    name = input('Enter name of this run:')
    
    
    
    #===========================================================================
    # Decide which gesture data shall be used for training
    #===========================================================================
    inputGestures = [0,1,2,3,4,5,6,7,8,9]
    
    #===========================================================================
    # Decide which target signals shall be used for training
    #===========================================================================
    usedGestures = [0,1,2,3,4,5,6,7,8,9]
    
    #===========================================================================
    # Concatenate data to create "more" training samples, 1 corresponds to no concatenations
    #===========================================================================
    concFactor = 1
    
    #===========================================================================
    # Add noise to the data, 0 corresponds to no noise. Noise above 2 has shown to weaken recognition
    #===========================================================================
    noiseFactor = 1
    
    #===========================================================================
    # Decide wether gestures shall be shuffled before training. If true, nFolds many 
    # pieces will be generated. Not every piece is garanteed to contain every gesture, so do not use too many.
    #===========================================================================
    shuffle = True
    nFolds = 4
    
    
    #===========================================================================
    # Function used to evaluate during cross validation. Possible functions are:
    # Evaluation.calc1MinusF1FromMaxApp (best working, used in thesis)
    # Oger.utils.nmse (normalised mean square error, tells nothing about classifier perfomance but works okay)
    # Evaluation.calcLevenshteinError (use the Levenshtein error, disadvantages are highlighted in thesis) 
    # Evaluation.calc1MinusF1FromInputSegment (use segmentation by supervised signal)
    #===========================================================================
    evaluationFunction = Evaluation.calc1MinusF1FromMaxApp
    
    #===========================================================================
    # Set this to true if another output neuron shall be added to represent "no gesture"
    #===========================================================================
    learnTreshold = False
    
    #===========================================================================
    # Use on of the optimisation dictionaries from the optDicts file
    #===========================================================================
    optDict = 'bestParas'
    
    #===========================================================================
    # Pick datasets to train on, and datasets to test on
    #===========================================================================
    inputFiles = ['nike','julian','nadja','line']
    testFiles = ['stephan']
    
    # If desired add a specific file to test on, e.g. randTestFiles = ['lana_0_0.npz']
    randTestFiles = []
    
    
    
    #===========================================================================
    # Setup project directory
    #===========================================================================
    now = datetime.datetime.now()
    resultsPath = getProjectPath()+'results/'
    pdfFileName = now.strftime("%Y-%m-%d-%H-%M")+'_'+name+'.pdf'
    pdfFilePath = resultsPath+'pdf/'+pdfFileName
    npzFileName = now.strftime("%Y-%m-%d-%H-%M")+'_'+name+'.npz'
    npzFilePath = resultsPath+'npz/'+npzFileName
    bestFlowPath = resultsPath+'nodes/'+now.strftime("%Y-%m-%d-%H-%M")+'_'+name+'.p'
    pp = PdfPages(pdfFilePath)
    
    
    #===========================================================================
    # Add labels for gestures
    #===========================================================================
    totalGestureNames = ['left','right','forward','backward','bounce up','bounce down','turn left','turn right','shake lr','shake ud', \
                         'tap 1','tap 2','tap 3','tap 4','tap 5','tap 6','no gesture']
    gestureNames = []
    for i in usedGestures:
        gestureNames.append(totalGestureNames[i])
    gestureNames.append('no gesture')
    
     
    
    
    
    
    
    #read datasets and add them to dataStep
    trainSets = []
    randTestSets = []
    dataStep = []
    
    for fileName in inputFiles:
        ind, t  = DataSet.createData(fileName, inputGestures,usedGestures)
        dataStep.append((ind,t))
        
    segs = splitBySignals(dataStep)

    #if desired shuffle and refraction the gestures
    if(shuffle):
        dataStep = shuffleDataStep(dataStep, nFolds)
    
    
    #if desired stretch testset
    newDataStep = []
    for ind, t in dataStep:
        
        indSets = []
        tSets = []
        for concer in range(concFactor):
            indSets.append(ind)
            tSets.append(t)
        ind = np.concatenate(indSets,0)
        t = np.concatenate(tSets,0)
        if noiseFactor > 0:
            for i in ind:
                i[0:3] = i[0:3]+np.random.normal(0,0.05 *noiseFactor)
                i[3:6] = i[3:6]+np.random.normal(0,0.5 * noiseFactor)
                i[6:9] = i[6:9]+np.random.normal(0,1.25 * noiseFactor)
        
        # if treshold shall be learned, another target signal needs to be added. 
        # No gestures signal is 1 if all other targets are 0.
        if learnTreshold:
            t = np.append(t,np.subtract(np.ones((t.shape[0],1)),np.max(t,1,None,True)*2),1)
        
        newDataStep.append((ind,t))
    dataStep = newDataStep
            
        
    # Training data must be input twice, for reservoir AND readout node
    data=[dataStep,dataStep]

    # Create single testfiles if requiered
    for iFile in randTestFiles:
        randTestSets.append(DataSet.createDataSetFromFile(iFile))
    







    #---------------------------------------------------------------------------------------------------#
    #--------------------------------------------GRIDSEARCH---------------------------------------------#
    #---------------------------------------------------------------------------------------------------#  

    reservoir = SparseNode()
    reservoir.updateInputScaling(dataStep) #adjust reservoir inputscaling to current trainset 
    readoutnode = Oger.nodes.RidgeRegressionNode()
    flow = mdp.Flow( [reservoir,readoutnode])
    
    

    #Create optDict
    resParams, readoutParams = OptDicts.getDicts(optDict)
    gridsearch_parameters = {reservoir:resParams,\
                             readoutnode:readoutParams}
    
    
    #Create optimizer
    opt = Oger.evaluation.Optimizer(gridsearch_parameters, evaluationFunction)
    
    
    #===========================================================================
    # Uncomment the following to lines to use single process training.
    #===========================================================================
    opt.scheduler = mdp.parallel.ProcessScheduler(n_processes=2, verbose=False)
    mdp.activate_extension("parallel")
    
    #Start gridsearch using nFolds and cross validation
    opt.grid_search(data, flow, n_folds=nFolds, cross_validate_function=Oger.evaluation.n_fold_random)
    

    
    #Plot minimum errors        
    Evaluation.plotMinErrors(opt.errors, opt.parameters, opt.parameter_ranges, pp)
    
    
    #Plot errorspace along thee axis:
    i = 0
    axisOne = -1
    axisTwo = -1
    axisThree = -1
    for node , param in opt.parameters:
        if param == 'spectral_radius':
            axisOne = i
        elif param == 'leak_rate':
            axisTwo = i
        elif param == 'ridge_param':
            axisThree = i
        i =i+1
    
    if axisThree != -1:
        if axisOne != -1:
            if axisTwo != -1:
                Evaluation.plotAlongAxisErrors(opt.errors, opt.parameters, opt.parameter_ranges, axisThree, axisOne, axisTwo, pp)
    
    
    
    
    #===========================================================================
    # Retrieve the best flow found during gridsearch, safe an untrained copy before training on all traindata
    #===========================================================================
    bestFlow = opt.get_optimal_flow(True)
    bestFlowUntrained = bestFlow.copy()
    bestFlow.train(data)
    
    
    #---------------------------------------------------------------------------------------------------#
    #---------------------------------------------TRAIN EVAL--------------------------------------------#
    #---------------------------------------------------------------------------------------------------# 

    print '#######################################################'
    print '                       TRAIN EVAL                      '
    print '#######################################################'
     
    
    nInputFiles = len(inputFiles)
    fig, axes = plt.subplots(nInputFiles, 1, sharex=True, figsize=(20,20))
    plt.tight_layout()
    plt.title('Prediction on training')  
    i = 0 
    trainCms = []
    trainPredicitions = []
    trainTargets = []
    trainF1s = []
    trainF1MaxApps=[]
    for row in axes:
        prediction = bestFlow([data[0][i][0]])
        t_target = data[0][i][1]
        
        if learnTreshold:
            learnedTreshold = prediction[:,-1]
            t_prediction = prediction[:,:-1]
            t_target = t_target[:,:-1]
        
        
        #visCalcConfusionFromMaxTargetSignal(prediction, t_target)
        row.set_title('Trainset '+str(i))
        row.plot(prediction)
        row.plot(np.atleast_2d(data[0][i][1]))
        pred, targ = Evaluation.calcInputSegmentSeries(prediction, t_target, 0.4, False)
        trainF1s.append(np.mean(sklearn.metrics.f1_score(targ,pred,average=None)))
        
        t_maxApp_prediction = Evaluation.calcMaxActivityPrediction(prediction,t_target,0.5,10)
        pred_MaxApp, targ_MaxApp = Evaluation.calcInputSegmentSeries(t_maxApp_prediction, t_target, 0.5)
        trainF1MaxApps.append(np.mean(sklearn.metrics.f1_score(targ_MaxApp,pred_MaxApp,average=None)))
        
        conf = sklearn.metrics.confusion_matrix(targ_MaxApp, pred_MaxApp)
        trainCms.append(conf)
        trainPredicitions.append(prediction)
        trainTargets.append(t_target)
        i = i+1
    #plt.plot(data[0][0][0])
    pp.savefig()
   
    
    for enum, trainCm in enumerate(trainCms):
        Evaluation.plot_confusion_matrix(trainCm, gestureNames, 'trainging set'+str(enum))
        pp.savefig()
        
    
    totalTrainCm = np.sum(np.concatenate(map(np.atleast_3d, trainCms),2),2)
    Evaluation.plot_confusion_matrix(totalTrainCm, gestureNames, 'Total training data')
    pp.savefig()
    
   
    totalTrainInputData = [x[0] for x in dataStep]
    totalTrainTargetData = [x[1] for x in dataStep]
    totalTrainInputData = np.concatenate(totalTrainInputData,0)
    totalTrainTargetData = np.concatenate(totalTrainTargetData,0)
    totalTrainPrediction = bestFlow(totalTrainInputData)
    tresholds, _, bestF1ScoreTreshold = Evaluation.calcTPFPForThresholds(totalTrainPrediction, totalTrainTargetData, 'Train Data Confusion - Target Treshold', False)
    pp.savefig()
    
    
    
    f1Cross = Oger.evaluation.validate(data, bestFlowUntrained, Evaluation.calc1MinusF1FromMaxApp, cross_validate_function=Oger.evaluation.leave_one_out)
    accuracyCross = Oger.evaluation.validate(data, bestFlowUntrained, Evaluation.calcAccuracyFromMaxApp, cross_validate_function=Oger.evaluation.leave_one_out)

    #---------------------------------------------------------------------------------------------------#
    #----------------------------------------------TESTING----------------------------------------------#
    #---------------------------------------------------------------------------------------------------#  
   
    print '#######################################################'
    print '                      TEST EVAL                        '
    print '#######################################################'
     
   
    confMatrices = []
    missClassifiedGestures = []
    f1Scores = []
    f1BestPossibleScores = []
    f1ppScores = []
    f1maxAppScores = []
    f1maxAppBestPossibleScores= []
    f1ScoreNames = []
    accuracies = []
    levs = []
    levs_pp = []
    
    #===========================================================================
    # Test on specific test files, if any are given
    #===========================================================================
    for set, setName in zip(randTestSets,randTestFiles):
        #set = DataSet.appendDataSets(set,DataSet.createDataSetFromFile('stephan_1_0.npz'))
        t_prediction = bestFlow([set.getDataForTraining(usedGestures, 2)[0]])
        t_target = set.getDataForTraining(usedGestures,2)[1]
        fig = plt.figure()
        fig.suptitle(setName)
        plt.clf()
        plt.subplot(211)
        plt.title('Prediction on test ' +setName)
        plt.plot(t_prediction)
        plt.plot(set.getDataForTraining(usedGestures,2)[1])
        plt.subplot(212)
        plt.title('Smoothed prediction')
        plt.plot(runningAverage(t_prediction, 10))
        plt.plot(set.getDataForTraining(usedGestures,2)[1])
        pp.savefig()
        print setName
        pred, targ = Evaluation.calcInputSegmentSeries(t_prediction, t_target, 0.4, False)
        cm = sklearn.metrics.confusion_matrix(targ, pred)
        f1,_ = Evaluation.calcF1ScoreFromConfusionMatrix(cm,True)
        confMatrices.append(cm)
        f1Scores.append(f1)
        f1ScoreNames.append(setName)
        f1ppScores.append(-1)
        Evaluation.plot_confusion_matrix(cm,gestureNames,setName + ' - full gesture ranking')
        pp.savefig()


    #===========================================================================
    # Test on testset
    #===========================================================================
    for iFile in testFiles:
        t_target,t_prediction, t_pp_prediction, t_maxApp_prediction, learnTreshold = EvaluateTestFile.evaluateTestFile(iFile,inputGestures,usedGestures, gestureNames, totalGestureNames, reservoir, bestFlow, tresholds, bestF1ScoreTreshold, shuffle, learnTreshold, f1Scores,f1BestPossibleScores, f1ppScores, f1maxAppScores, f1maxAppBestPossibleScores, f1ScoreNames,accuracies, levs, levs_pp, pp, confMatrices)
        
    
    totalCm = confMatrices[0]
    for cm in confMatrices[1:]:
        totalCm = totalCm+cm
    Evaluation.plot_confusion_matrix(totalCm,gestureNames,'total test confusion')    
    pp.savefig()
 
    
    

    
    pp.close();  



    #---------------------------------------------------------------------------------------------------#
    #-----------------------------------------------REPORT----------------------------------------------#
    #---------------------------------------------------------------------------------------------------#  

    #===============================================================================
    # Write results of this run into the resport .csv 
    #===============================================================================
    inFiles = inputFiles
    result = [str(now),name,inputFiles,testFiles,opt.loss_function, \
              'TrainError',str(opt.get_minimal_error()[0]), 'Train F1', np.mean(trainF1s),'Train F1 MaxApp',np.mean(f1Cross),'train accuracies',np.mean(accuracyCross),\
              'meanF1Score', f1Scores, 'bestPossF1',f1BestPossibleScores,'meanPPF1Score',f1ppScores,'maxAppF1Score',f1maxAppScores,'learnedTreshold',bestF1ScoreTreshold,'bestPossMaxAppF1',f1maxAppBestPossibleScores,'accuracies',accuracies,\
              'Levenshtein',levs,'Levenshtein_maxApp',levs_pp]
    

    result.extend(['inputGestures',inputGestures])
    result.extend(['usedGestures',usedGestures])
    result.extend(['stretchFactor',concFactor])
    result.extend(['noiseFactor',noiseFactor])
    
      
    minErrDict = opt.get_minimal_error()[1]
    sparseDict = minErrDict.get(reservoir)
    ridgeDict = minErrDict.get(readoutnode)
    
    for para in ['_instance','inputSignals','useNormalized','input_scaling','output_dim','spectral_radius','useSparse','leak_rate','ridge_param']:
        result.append(para)
        if sparseDict is not None and sparseDict.has_key(para):
            result.append(sparseDict.get(para)) 
        elif ridgeDict is not None and ridgeDict.has_key(para):
            result.append(ridgeDict.get(para)) 
        else:
            result.append('')
     
     
     
    print 'w outputdim:' + str(bestFlow[0].w.shape)
     
    
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
    


    
    result.append('=HYPERLINK(\"'+pdfFilePath+'\")')
    result.append('=HYPERLINK(\"'+npzFilePath+'\")')
    result.append('=HYPERLINK(\"'+bestFlowPath+'\")')
    result.extend(['optDict',optDict])
    
    if name != 'test':
        writeToReportFile(result)
        np.savez(npzFilePath,errors=opt.errors,params=opt.parameters,paraRanges=opt.parameter_ranges,randTestFiles=randTestFiles,\
                 confMatrices=confMatrices, \
                 testFileList=randTestFiles,\
                 f1Scores=f1Scores,\
                 f1ScoreNames=f1ScoreNames,\
                 bestRes_w_in=bestFlow[0].w_in, \
                 bestRes_w=bestFlow[0].w, \
                 t_prediction=t_prediction,\
                 t_target=t_target\
                 )
        bestFlow.save(bestFlowPath)
        

    