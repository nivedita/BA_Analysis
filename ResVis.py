
import numpy as np
import matplotlib


import matplotlib.pyplot as plt

from Oger.nodes.reservoir_nodes import ReservoirNode
from sklearn.manifold.t_sne import TSNE
from sklearn.manifold.spectral_embedding_ import SpectralEmbedding
from sklearn.manifold.mds import MDS
import time
from numpy import argmax
from matplotlib.backends.backend_pdf import PdfPages
from Main import getProjectPath


def execute(res, pos, signal, inputSignals ,targets, max,  pp, title='' ):
    signal = np.atleast_2d(signal) * inputSignals
    res.execute(np.atleast_2d(signal[0,:]))
    fig = plt.figure(figsize=(20,20),title=title)
    neuronFig = plt.subplot2grid((2,2), (0,0), 2,1)
    mapable = neuronFig.scatter(pos[:,0],pos[:,1],c=res.states,cmap='BrBG',vmin=-1, vmax=1,marker='o',s=200)
    fig.colorbar(mapable)
    inputFig = fig.add_subplot(222)
    targetFig = fig.add_subplot(224)

    inputFig.set_ylim([-2, 2])
    
    res.execute(signal)

    #print 'total activity:'+str(np.sum(np.abs(res.states)))
    plotSeq= range(30,90)
    plotSeq.extend(range(3000,3060))
    for i in plotSeq:
        #print res.states[i,0:10]
        #print (np.atleast_2d(signal)*inputSignals)[i,:]
        
        #res.execute(np.atleast_2d(signal[i,:]))
        states = res.states[i,:]
        
        neuronFig.cla()
        targetFig.cla()
        
        
        neuronFig.set_title('Reservoir acitivity')
        neuronFig.scatter(pos[:,0],pos[:,1],c=states,cmap='BrBG',vmin=-max, vmax=max,marker='o',s=200)
        
        #print states
        if targets is not None:
            if np.max(targets[i,:]) == 1: 
                fig.patch.set_facecolor('white')
                targetFig.text(np.min([30,i]), targetFig.get_ylim()[1]-1, 'gesture ' + str(np.argmax(targets[i,:])), fontsize=20)
            else: 
                fig.patch.set_facecolor('grey')
                targetFig.text(np.min([30,i]), targetFig.get_ylim()[1]-1, 'no gesture', fontsize=20)
                
            #print str(i) + '    total activity:    '+str(np.sum(np.abs(states)))+'    '+ str(targets[i,:])
        else:
            #print str(i) + '    total activity:    '+str(np.sum(np.abs(states)))
            pass
        #plt.colorbar()
        startInd = np.max([i-30,0])
        endInd =  np.min([i+30,len(signal)])
        
        
        inputFig.cla()
        inputFig.set_title('Input signal')
        inputFig.plot(signal[startInd:endInd,:])
        inputFig.plot([np.min([30,i]),np.min([30,i])],[-10,10],linestyle='-')
        inputFig.set_ylim([-15, 15])
        inputFig.set_xlim([0, 60])
        
        targetFig.set_title('Target signal')
        targetFig.plot(targets[startInd:endInd,:])
        targetFig.plot([np.min([30,i]),np.min([30,i])],[-2,2],linestyle='-')
        targetFig.set_ylim([-2, 2])
        targetFig.set_xlim([0, 60])
        
        
        
        if i % 40 == 0:
            #plt.waitforbuttonpress()
            pass
        
        plt.draw()
        #time.sleep(0.01)
        plt.pause(0.0001) 
        pp.savefig()
    

def plotMDS():
    res = ReservoirNode(input_dim=2,output_dim=40, spectral_radius=0.999999, reset_states=False)
    print
    print '--------------------------------------------------'
    print
    
    
    res.initialize()     
    
    print res.spectral_radius
    clf = MDS(2,metric='precomputed')
    
    X = np.copy(res.w)
    m_X = np.fliplr(np.flipud(X))
    double = np.append(np.atleast_3d(X),np.atleast_3d(m_X),2)
    X = np.mean(double, 2)
    
    X = np.abs(X)
    pos = clf.fit_transform(X)
    
    
    #,[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]
    res.execute(np.array([[10,0]]))
    plt.figure(1)
    plt.clf
    plt.scatter(pos[:,0],pos[:,1],c=np.abs(res.states),cmap='Reds',vmin=0, vmax=1,marker='o')
    plt.colorbar()
    print 'total activity:'+str(np.sum(np.abs(res.states)))
    plt.waitforbuttonpress()
    for i in range(30):
        res.execute(np.array([[0,0]]))
        states = res.states[0,:]
        #print states
        
        print 'total activity:'+str(np.sum(np.abs(states)))
        plt.scatter(pos[:,0],pos[:,1],c=np.abs(states),cmap='Reds',vmin=0, vmax=1,marker='o')
        plt.draw()
        plt.waitforbuttonpress()
    
    plt.show()


    
def plotRes(res,input_signal,targets=None,artTrainingData=False):
#if __name__ == '__main__':    
    print
    print '--------------------------------------------------'
    print
    
    resultsPath = getProjectPath()+'results/'
    pdfFileName = 'resVis.pdf'
    pdfFilePath = resultsPath+'pdf/'+pdfFileName
    pp = PdfPages(pdfFilePath)
    
    signal = np.copy(input_signal)
    
    if artTrainingData:
        trainingData = np.zeros((signal.shape[1]*100,signal.shape[1]))
        for i in range(0,signal.shape[1]):
            for j in range(0,10):
                trainingData[i*100+j*10][i] = 1
    else:
        trainingData = signal
            
    
    #signal[:,3:9]=0
    

    #signal = np.concatenate([signal, signal, signal],0)
    res.execute(trainingData)
    
    #print 'total activity:'+str(np.sum(np.abs(res.states),1))
    
    data = np.copy(res.states).T
    max=np.max(np.abs(data))
    
    #data = data-np.mean(data,0)
    #data = data/np.std(data, 0)
    
    pos = TSNE(2).fit_transform(data)
    
    
    


    

    #plt.colorbar()
    #plt.waitforbuttonpress()
    execute(res, pos, signal, np.array([1,1,1,1,1,1,1,1,1]), targets, max, pp, 'All Signals')
    execute(res, pos, signal, np.array([1,1,1,0,0,0,0,0,0]), targets, max, pp, 'Only fused')
    execute(res, pos, signal, np.array([0,0,0,1,1,1,0,0,0]), targets, max, pp, 'Only gyro')
    execute(res, pos, signal, np.array([0,0,0,0,0,0,1,1,1]), targets, max, pp, 'Only lin')
    print data.shape
    pp.close()
    
    
