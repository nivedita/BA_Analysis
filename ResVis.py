
import numpy as np
import matplotlib


import matplotlib.pyplot as plt

from Oger.nodes.reservoir_nodes import ReservoirNode
from sklearn.manifold.t_sne import TSNE
from sklearn.manifold.spectral_embedding_ import SpectralEmbedding
from sklearn.manifold.mds import MDS
import time


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


    
def plotRes(res,signal,targets=None):
    
    print
    print '--------------------------------------------------'
    print
    

    
    

    #signal = np.concatenate([signal, signal, signal],0)
    res.execute(signal)
    
    print 'total activity:'+str(np.sum(np.abs(res.states),1))
    
    data = np.copy(res.states).T
    max=np.max(np.abs(data))
    
    #data = data-np.mean(data,0)
    #data = data/np.std(data, 0)
    
    pos = TSNE(2).fit_transform(data)
    
    
    
    res.execute(np.atleast_2d(signal[0,:]))
    fig = plt.figure(1)
    plt.clf
    plt.scatter(pos[:,0],pos[:,1],c=res.states,cmap='BrBG',vmin=-1, vmax=1,marker='o',s=200)
    plt.colorbar()
    plt.waitforbuttonpress()
    print 'total activity:'+str(np.sum(np.abs(res.states)))
    for i in range(1,100):
        res.execute(np.atleast_2d(signal[i,:]))
        states = res.states[0,:]
        #print states
        if targets is not None:
            if np.max(targets[i,:]) == 1: 
                fig.patch.set_facecolor('white')
                plt.text(20, 20, 'gesture', fontsize=12)
            else: 
                fig.patch.set_facecolor('grey')
                plt.title('')
            print str(i) + '    total activity:    '+str(np.sum(np.abs(states)))+'    '+ str(targets[i,:])
        else:
            print str(i) + '    total activity:    '+str(np.sum(np.abs(states)))
        plt.clf()
        plt.scatter(pos[:,0],pos[:,1],c=states,cmap='BrBG',vmin=-max, vmax=max,marker='o',s=200)
        plt.colorbar()
        
        if i % 10 == 0:
            plt.waitforbuttonpress()
        
        plt.draw()
        time.sleep(0.1)
        plt.pause(0.0001) 
    plt.show()
    
    print data.shape
    
    
    

if __name__ == '__main__':
    res = ReservoirNode(input_dim=2,output_dim=400, spectral_radius=0.999999, reset_states=False)
        
    signal = np.zeros((40,2))
    signal[0,0] = 5
    signal[39,1] = 5
    #plotRes(res, signal)
    