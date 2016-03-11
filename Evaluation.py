import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from gettext import ngettext

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
    valsP, indsP = mergePredictions(input_signal, True)
    valsT, indsT = mergePredictions(target_signal, True)
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


def calc1MinusF1Average(input_signal,target_signal):
    cm, _ = calcConfusionMatrix(input_signal, target_signal)
    f1Scores, _ = calcF1ScoreFromConfusionMatrix(cm)
    return 1-np.mean(f1Scores)
    
    


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