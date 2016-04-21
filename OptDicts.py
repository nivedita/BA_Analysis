import numpy as np
def getDicts(name):
    
    dict = {
    'bigRunLevDict':(   {'useSparse':[True,False], \
               'inputSignals':['FGA'], \
               'useNormalized':[2], \
               'leak_rate':np.arange(0.1,1,0.1), \
               'spectral_radius':np.arange(0.1,1.5,0.3), \
               'output_dim':[400], \
               'input_scaling':np.arange(1,15,3), \
               '_instance':range(5)},\
               {'ridge_param':np.arange(0,7,2)} ),\
    

    #### dict fuer conc and noise
    'concAndNoiseOld':({'useSparse':[True], \
                                        'inputSignals':['FGA'], \
                                        'useNormalized':[2], \
                                        'leak_rate':[0.2], \
                                        'spectral_radius':[0.9], \
                                        'output_dim':[400], \
                                         'input_scaling':[10], \
                                        '_instance':range(2)}, \
                             {'ridge_param':[4]} ),\
            
    'concAndNoise':({'useSparse':[True], \
                                        'inputSignals':['FGA'], \
                                        'useNormalized':[2], \
                                        'leak_rate':[0.2], \
                                        'spectral_radius':[0.9], \
                                        'output_dim':[400], \
                                         'input_scaling':np.arange(10,15,10), \
                                        '_instance':range(4)}, \
                             {'ridge_param':[4]}),\
    
    #best paras
    'bestParas':({'useSparse':[True], \
                                        'inputSignals':['FGA'], \
                                        'useNormalized':[2], \
                                        'leak_rate':[0.3], \
                                        'spectral_radius':[1], \
                                        'output_dim':[400], \
                                         'input_scaling':[13], \
                                        '_instance':range(5)}, \
                             {'ridge_param':[0.01]}),
    
    #best paras
    'test':({'useSparse':[True], \
                                        'inputSignals':['FGA'], \
                                        'useNormalized':[2], \
                                        'leak_rate':[0.3], \
                                        'spectral_radius':[1], \
                                        'output_dim':[400], \
                                         'input_scaling':[100], \
                                        '_instance':range(4)}, \
                             {'ridge_param':[0.01]}),
            
    'noiseAndConc':({'useSparse':[True], \
                                        'inputSignals':['FGA'], \
                                        'useNormalized':[2], \
                                        'leak_rate':[0.2], \
                                        'spectral_radius':[0.9], \
                                        'output_dim':[400], \
                                         'input_scaling':np.arange(10,15,10), \
                                        '_instance':range(4)}, \
                             {'ridge_param':[4]})     
    }
    
    return dict.get(name)
