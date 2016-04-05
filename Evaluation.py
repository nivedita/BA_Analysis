import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from gettext import ngettext
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal._peak_finding import argrelextrema
import sklearn
import sklearn.metrics
from sklearn.metrics.ranking import roc_curve, auc
from scipy import interp



def mergePredictions(predictions, addTreshold=False, treshold=0.0, plot=False):
    if addTreshold:
        predictions = np.append(predictions, np.ones((len(predictions),1))*treshold, 1)
    vals = np.max(predictions,1)
    inds = np.argmax(predictions, 1)
    
    if plot:
        plt.figure()
        plt.plot(predictions, color='grey')
        plt.plot(vals, color='r')
        plt.plot(inds, color='g')
    
    
    return vals, inds


def calcConfusionMatrix(input_signal,target_signal):
    nGestures = len(target_signal[0])
    valsP, indsP = mergePredictions(input_signal, True, 0.5)
    valsT, indsT = mergePredictions(target_signal, True, 0.5)
    changesP = np.where(indsP[:-1] != indsP[1:])[0] + 1  # indexes where predicted gesture changes 
    changesT = np.where(indsT[:-1] != indsT[1:])[0] + 1  # indexes where actual gesture changes
    
    
    detections = []
    classifiedGestures = [[[] for x in range(nGestures+1)] for x in range(nGestures+1)] 
    # fuer jedes segment, auch wenn gerade keine gestge stattfindet
    lastInd = 0
    for ind in changesT:  
        cur_valsP = valsP[lastInd:ind]
        cur_indsP = indsP[lastInd:ind]
        cur_valsT = valsT[lastInd:ind]
        cur_indsT = indsT[lastInd:ind]

        occurences = np.bincount(cur_indsP, None, nGestures+1) # +1 wegen "keine geste"
        detectedGesture = np.argmax(occurences)
        actualGesture = cur_indsT[0]
        
        classifiedGestures[actualGesture][detectedGesture].append((lastInd,ind))
        detections.append((detectedGesture,actualGesture))
        lastInd = ind
    
    confusionMatrix = np.zeros((nGestures+1,nGestures+1)) # +1 wegen "keine geste"
    for det, act in detections:
        confusionMatrix[act][det] = confusionMatrix[act][det] + 1
    return confusionMatrix, classifiedGestures
    
    
def calcF1ScoreFromConfusionMatrix(cm, replaceNan = True):
    f1Scores = np.zeros((len(cm),1))
    for i in range(0,len(cm)):
        tp = cm[i][i]
        fp = np.sum(cm[:,i])-tp
        fn = np.sum(cm[i,:])-tp
        f1Scores[i]= (2.*tp)/(2.*tp+ fn+ fp)
    occurences = np.sum(cm,1)
    #replace nan
    if replaceNan:
        mean = np.mean(f1Scores[np.invert(np.isnan(f1Scores))])
        f1Scores[np.isnan(f1Scores)]=mean
    return f1Scores, occurences


def calc1MinusF1Average(input_signal,target_signal,noSilence = False):
    cm, _ = calcConfusionMatrix(input_signal, target_signal)
    f1Scores, _ = calcF1ScoreFromConfusionMatrix(cm)
    if noSilence:
        return 1-np.mean(f1Scores[:-1])
    return 1-np.mean(f1Scores)
    
    
def calcFloatingAverage(input_signal,target_signal):
    offset = 5
    floatingSum = np.zeros(input_signal.shape)
    for i in range(offset,input_signal.shape[0]):
        floatingSum[i] = np.sum(input_signal[i-offset:i,:],0)
    return floatingSum
    
def calcF1OverFloatingAverage(input_signal,target_signal):
    return calc1MinusF1Average(calcFloatingAverage(input_signal, target_signal),target_signal)

def plot_confusion_matrix(cm, gestures=None,title='Confusion matrix', cmap=cm.Blues):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    if gestures is not None:
        tick_marks = np.arange(len(gestures))
        plt.xticks(tick_marks, gestures, rotation=45)
        plt.yticks(tick_marks, gestures)
    

    ind_array = np.arange(0, len(cm), 1)
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = str(cm[y_val,x_val])
        plt.text(x_val, y_val, c, va='center', ha='center')
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

###
### counts max of last n steps
###
def createMaxTargetSignal(t_prediction, treshold):
    filterLength = 6
    t_max = np.zeros((t_prediction.shape[0],1))
    t_prediction = addTresholdSignal(t_prediction, treshold)
    for i in range(1,filterLength):
        t_max[i] = np.argmax(np.bincount(np.argmax(t_prediction[0:i,:], 1)))
    for i in range(filterLength,t_prediction.shape[0]):
        t_max[i] = np.argmax(np.bincount(np.argmax(t_prediction[i-filterLength:i,:], 1)))
    return t_max
    
def addTresholdSignal(prediction, treshold):
    return np.append(prediction, np.ones((prediction.shape[0],1))*treshold, 1)

def addNoGestureSignal(target):
    inds = np.max(target,1)==0
    inds = np.atleast_2d(inds.astype('int')).T
    return np.append(target, inds, 1)
    

def calc1MinusConfusionFromMaxTargetSignal(input_signal,target_signal, vis=False):
    treshold = 0.4
    maxPred= createMaxTargetSignal(input_signal,treshold)
    maxTarg= createMaxTargetSignal(target_signal, 0.9)
    confMatrix = sklearn.metrics.confusion_matrix(maxTarg, maxPred,None)
    f1scores = sklearn.metrics.f1_score(maxTarg,maxPred,average=None)
    #print f1scores
    f1score = np.mean(f1scores)
    
    if vis:
        plt.figure()
        plt.plot(maxPred)
        plt.plot(maxTarg)
        plt.plot(input_signal)
        plot_confusion_matrix(confMatrix)
        #print confMatrix
    #print f1score
    return 1-f1score
        
def visCalcConfusionFromMaxTargetSignal(input_signal,target_signal, treshold=0.4):
    maxPred= createMaxTargetSignal(input_signal,treshold)
    maxTarg= createMaxTargetSignal(target_signal, np.max([0.001,treshold]))
    confMatrix = sklearn.metrics.confusion_matrix(maxTarg, maxPred,None)
    f1scores = sklearn.metrics.f1_score(maxTarg,maxPred,average=None)
    f1score = np.mean(f1scores)
    report = sklearn.metrics.classification_report(maxTarg.astype('int'), maxPred.astype('int'))
    #print report
    return confMatrix, f1scores, f1score


def countTargetAndPredictedSignalsPerGesture(input_signal,target_signal):
    results = []
    for sigNr in range(0,len(input_signal[0])):
        cur_input_signal = input_signal[:,sigNr]
        cur_target_signal = target_signal[:,sigNr]
        nDataPoints = len(cur_input_signal)
        treshold = 0.5
        n_totalTarget = 0
        n_totalPredicted = 0
        i = 0
        while i < nDataPoints:
            if cur_target_signal[i] == 1.0:
                n_totalTarget = n_totalTarget+1
            while i+1 < nDataPoints and cur_target_signal[i] == 1 and cur_target_signal[i+1] == 1 :
                i = i+1
            i = i+1
        i = 0
        while i < nDataPoints:
            if cur_input_signal[i] > treshold:
                n_totalPredicted= n_totalPredicted+1
            while i+1 < nDataPoints and cur_input_signal[i] > treshold and cur_input_signal[i+1] > treshold:
                i = i+1
            i = i+1
        results.append((n_totalTarget, n_totalPredicted))
    return results


def plotMinErrorsToFIle(opt):
    pdfFileName = 'minErrors.pdf'
    pdfFilePath = getProjectPath()+'results/pdf/'+pdfFileName
    pp = PdfPages(pdfFilePath)
    plotMinErrors(opt.errors, opt.parameters, opt.parameter_ranges, pp)
    pp.close()

   
def plotMinErrors(errs, params,ranges,pp):
    minVal = np.min(errs)
    min_ind = np.unravel_index(errs.argmin(), errs.shape)
    for i in range(0,len(min_ind)):
        for j in range(i,len(min_ind)):
            if(j != i and errs.shape[i] > 1 and errs.shape[j] > 1 and \
                params[i][1] != '_instance' and params[j][1] != '_instance' ):
                minAxes = range(0,len(min_ind))
                minAxes.remove(i)
                minAxes.remove(j)
                mins = np.min(errs,tuple(minAxes))
                plt.figure()
                plt.imshow(mins, interpolation='nearest',cmap='Blues',vmin=minVal, vmax=1)
                plt.xlabel(params[j][1])
                plt.ylabel(params[i][1])
                
                plt.colorbar()
                if ranges is not None:
                    tick_marks = np.arange(len(mins[0]))
                    plt.xticks(tick_marks, ranges[j], rotation=45)
                    tick_marks = np.arange(len(mins))
                    plt.yticks(tick_marks, ranges[i])
                plt.tight_layout()
                
                if pp is not None:
                    pp.savefig()
                #plot_confusion_matrix(cm, gestures, title, cmap)
        #TODO:plot all dims
    

def plotAlongAxisErrors(errs, params,ranges,plotAxis, xAxis, yAxis, pp):
    minVal = np.min(errs)
    min_ind = np.unravel_index(errs.argmin(), errs.shape)
    
    nParams = len(params)
    if plotAxis >= nParams or xAxis >= nParams or yAxis >= nParams or plotAxis is None or xAxis is None or yAxis is None:
        return
    minAxes = range(0,len(params))
    minAxes.remove(plotAxis)
    minAxes.remove(xAxis)
    minAxes.remove(yAxis)
    totalMins = np.min(errs,tuple(minAxes),None,True)
    print totalMins.shape
    for i in range(0, len(ranges[plotAxis])):
        plt.figure()
        plt.title(params[plotAxis][1] + ' = ' + str(ranges[plotAxis][i]))
        mins = np.delete(totalMins, range(0,i), plotAxis)
        mins = np.delete(mins, range(1,100),plotAxis)
        mins = np.atleast_2d(np.squeeze(mins))
        plt.imshow(mins, interpolation='nearest',cmap='Blues',vmin=minVal, vmax=1)
        plt.xlabel(params[xAxis][1])
        plt.ylabel(params[yAxis][1])
                
        plt.colorbar()
        if ranges is not None:
            tick_marks = np.arange(len(mins[0]))
            plt.xticks(tick_marks, ranges[xAxis], rotation=45)
            tick_marks = np.arange(len(mins))
            plt.yticks(tick_marks, ranges[yAxis])
        plt.tight_layout()
                
        if pp is not None:
            pp.savefig()
                
    
def showROC(prediction, target):
    nGestures = target.shape[1]

    n_classes = nGestures
    y_test = target
    y_score = prediction
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    
    
    ##############################################################################
    # Plot ROC curves for the multiclass problem
    
    # Compute macro-average ROC curve and ROC area
    
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             linewidth=2)
    
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             linewidth=2)
    
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i],label='ROC curve of class {0} (area = {1:0.2f})'
                                       ''.format(i, roc_auc[i]))
        

    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

def getMinima(errs, nr=-1):
    
    inds = argrelextrema(errs, np.less,order=1, mode='wrap')
    indTable = np.zeros((len(inds[0]),len(errs.shape)))
    for i in range(0,len(inds)):
        indTable[:,i] = inds[i]
    if nr == -1:
        return indTable
    else:
        return indTable[nr,:]
    
    
    
def calcInputSegmentSeries(prediction, target, treshold, plot=False):
    inds = [0]
    prediction = addTresholdSignal(prediction, treshold)
    target = addNoGestureSignal(target)
    predictionInt = np.argmax(prediction, 1)
    targetInt = np.argmax(target,1)
    for i in range(1,len(predictionInt)):
        if predictionInt[i-1] != predictionInt[i]:
            inds.append(i)
    inds.append(len(prediction)-1)
    
    if plot :
        plt.figure()
        cmap = mpl.cm.gist_earth
        for i in range(prediction.shape[1]):
            plt.plot(prediction[:,i],c=cmap(float(i)/(prediction.shape[1])))
        
        lastI = 0
        for i in inds:
            #plt.plot([i,i],[-2,2], c='black')
            x = np.arange(lastI,i+1)
            y1 = 0
            y2 = prediction[x,predictionInt[lastI]]
            #print predictionInt[i], prediction.shape[1], float(predictionInt[i]) / prediction.shape[1]
            plt.fill_between(x, y1, y2, facecolor=cmap(float(predictionInt[lastI])/(prediction.shape[1])), alpha=0.5)
            lastI = i
        plt.plot(target)
    
    mapped = np.zeros(targetInt.shape)
    
    segmentPredicted = np.zeros((len(inds)-1,1))
    segmentTarget = np.zeros((len(inds)-1,1))
    mappedAway = 0
    falsePositives = 0
    for i in range(1,len(inds)):
        start = inds[i-1]
        end = inds[i]
        targetSegment = targetInt[start:end]
        predictedClass = predictionInt[start]
        
        segmentPredicted[i-1]=predictedClass
        
        print 'target and pred',targetSegment, predictedClass        
        
        #check for tp case
        tpInds = np.add(np.where(targetSegment==predictedClass),start)
        print tpInds, tpInds.size
        if tpInds.size==0 : #predicted class not found in segment
            print 'false positive',start,end
            falsePositives += 1
            segmentTarget[i-1]=np.argmax(np.bincount(targetSegment))
        elif np.max(mapped[tpInds])!=0:   #predicted class found in segment but mapped away
            print 'mapped away'
            mappedAway += 1
            segmentTarget[i-1]=prediction.shape[1]-1
        else:
            segmentTarget[i-1]=predictedClass            
            print 'tp'
            #wenn true positive gfunden wurde, entferne das target signal
            j = start
            while targetInt[j] == predictedClass: #wenn singal am anfang is muss man zurueck laufen
                j = j-1
            while targetInt[j] != predictedClass:
                j = j+1
            startDel = j
            while j < len(targetInt) and targetInt[j] == predictedClass:
                j = j+1
            endDel = j
            mapped[startDel:endDel] = 1 #this singal is mapped
            print mapped
        
    print 'mapped away',mappedAway
    print 'fale positive',falsePositives   
           
    ######################################################################
    #######TODO handling wenn langes no gesture segment
    ###################################################################### 

    print np.array(zip(segmentPredicted,segmentTarget))
        
