import numpy as np
def getDicts(name):
    
    dict = {
            
    'test':({'useSparse':[True], \
            'connectivity':[0.2], \
            'inputSignals':['FGA'], \
            'useNormalized':[2], \
            'leak_rate':[0.3], \
            'spectral_radius':[1], \
            'output_dim':[50], \
            'input_scaling':[10], \
            '_instance':range(3)}, \
            {'ridge_param':[0.0001]}),
            
    'bestParas2':({'useSparse':[True], \
            'connectivity':[0.1], \
            'inputSignals':['FGA'], \
            'useNormalized':[2], \
            'leak_rate':[0.3], \
            'spectral_radius':[0.7], \
            'output_dim':[400], \
            'input_scaling':[15], \
            '_instance':range(6)}, \
            {'ridge_param':[0.01]}), \
    
    'inputScaleNormConnect':({'useSparse':[True], \
            'connectivity':[0.1, 0.25, 0.5, 0.75, 1], \
            'inputSignals':['FGA'], \
            'useNormalized':[0,1,2], \
            'leak_rate':[0.3], \
            'spectral_radius':[1], \
            'output_dim':[400], \
            'input_scaling':[0.25,0.5,1,2,4,8], \
            '_instance':range(2)}, \
            {'ridge_param':[0.01]}), \
            
    'bigRunLevDict':({'useSparse':[True, False], \
               'inputSignals':['FGA'], \
               'useNormalized':[2], \
               'leak_rate':np.arange(0.1, 1, 0.1), \
               'spectral_radius':np.arange(0.1, 1.5, 0.3), \
               'output_dim':[400], \
               'input_scaling':np.arange(1, 15, 3), \
               '_instance':range(5)}, \
               {'ridge_param':np.arange(0, 7, 2)}), \
    

    #### dict fuer conc and noise
    'concAndNoiseOld':({'useSparse':[True], \
                                        'inputSignals':['FGA'], \
                                        'useNormalized':[2], \
                                        'leak_rate':[0.2], \
                                        'spectral_radius':[0.9], \
                                        'output_dim':[400], \
                                         'input_scaling':[10], \
                                        '_instance':range(2)}, \
                             {'ridge_param':[4]}), \
            
    'concAndNoise':({'useSparse':[True], \
                                        'inputSignals':['FGA'], \
                                        'useNormalized':[2], \
                                        'leak_rate':[0.2], \
                                        'spectral_radius':[0.9], \
                                        'output_dim':[400], \
                                         'input_scaling':np.arange(10, 15, 10), \
                                        '_instance':range(4)}, \
                             {'ridge_param':[4]}), \
    
    # best paras
    'bestParas':({'useSparse':[True], \
                                        'inputSignals':['FGA'], \
                                        'useNormalized':[2], \
                                        'leak_rate':[0.3], \
                                        'spectral_radius':[1], \
                                        'output_dim':[400], \
                                         'input_scaling':[13], \
                                        '_instance':range(5)}, \
                             {'ridge_param':[0.01]}),
    


    'influenceInputNormalisationVsScaling':({'useSparse':[True], \
                                        'inputSignals':['FGA'], \
                                        'useNormalized':[0, 1, 2], \
                                        'leak_rate':[0.3], \
                                        'spectral_radius':[1], \
                                        'output_dim':[400], \
                                         'input_scaling':[0.125, 0.25, 0.5, 1, 2, 4, 8, 16], \
                                        '_instance':range(4)}, \
                             {'ridge_param':[0.01]}),
            
    'noiseAndConc':({'useSparse':[True], \
                                        'inputSignals':['FGA'], \
                                        'useNormalized':[2], \
                                        'leak_rate':[0.2], \
                                        'spectral_radius':[0.9], \
                                        'output_dim':[400], \
                                         'input_scaling':np.arange(10, 15, 10), \
                                        '_instance':range(4)}, \
                             {'ridge_param':[4]})     
    }
    
    return dict.get(name)
