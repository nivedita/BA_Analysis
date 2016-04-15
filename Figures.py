import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm
from scipy.signal.windows import gaussian


def createTargetShapeDelayFigure():
    gestureLen = 20
    gestureSig = np.concatenate([np.zeros((10,3)),np.random.normal(size=(gestureLen,3)),np.zeros((10,3))],0)
    target = np.concatenate([np.zeros((10,1)),np.ones((gestureLen,1)),np.zeros((10,1))],0)
    target_gaus = np.concatenate([np.zeros((5,1)),np.atleast_2d(gaussian(gestureLen+10,5)).T,np.zeros((5,1))],0)
    target_delayed = np.concatenate([np.zeros((28,1)),np.ones((5,1)),np.zeros((7,1))],0)
    
    fig, ax = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(20,5))
    plt.ylim(-5,5)
    for axn in ax: 
        axn.plot(gestureSig,label='input signal')
        axn.plot([0,40],[0,0],c='black',linewidth=1)
    ax[0].plot(target,label='target',c='red',linewidth=2)
    ax[0].fill_between(np.arange(0,40),0,target.squeeze(),facecolor='red',alpha=0.5)
    ax[0].set_title('(a)')
    ax[1].plot(target_gaus,label='target',c='red',linewidth=2)
    ax[1].fill_between(np.arange(0,40),0,target_gaus.squeeze(),facecolor='red',alpha=0.5)
    ax[1].set_title('(b)')
    ax[2].plot(target_delayed,label='target',c='red',linewidth=2)
    ax[2].fill_between(np.arange(0,40),0,target_delayed.squeeze(),facecolor='red',alpha=0.5)
    ax[2].set_title('(c)')
    #plt.legend(bbox_to_anchor=(1., 1.05), loc=1, borderaxespad=0.)
    plt.tight_layout()
    projectPath = 'C:\Users\Steve\Documents\Uni\BAThesis\\src\\targetShapeDelay2.pdf'
    pp = PdfPages(projectPath)
    pp.savefig()
    pp.close()

if __name__ == '__main__':
    matplotlib.rcParams.update({'font.size': 20})
    