import numpy as np
import pandas as pd
import warnings

from sklearn.model_selection import train_test_split 
np.set_printoptions(suppress=True)

def load(qpo:str, context, context_type:str, rebin=None): 
    import pandas as pd 
    from collections import Counter   
    
    qpo_df = pd.read_csv(qpo)
    qpo_features = list(qpo_df.drop(columns=['observation_ID']))

    max_simultaneous_qpos = Counter(qpo_df['observation_ID']).most_common(1)[0][1]

    context_df = pd.read_csv(context)
    context_features = []

    observation_IDs = []

    context_tensor = None 
    max_length = max_simultaneous_qpos*len(qpo_features)
    qpo_tensor = np.array([]).reshape(0,max_length)

    train_test_indices = []

    spectral_ranges = None 
    spectral_centers = None 
    context_is_spectrum = False
    
    observation_IDs = list(context_df['observation_ID'])
    context_df = context_df.drop(columns=['observation_ID'])
    context_features = list(context_df)
    context_tensor = np.array(context_df)

    if context_type=='spectrum': 
        context_is_spectrum = True 
        
        spectral_ranges = []
        spectral_centers = []

        for col in context_features: 
            col_list = col.split('_')
            spectral_centers.append(float(col_list[0]))
            col_list = col_list[1].split('-')
            spectral_ranges.append((col_list[0], col_list[1]))

        if rebin is not None: 

            from scipy.stats import binned_statistic
            
            lows = [i[0] for i in spectral_ranges]
            highs = [i[1] for i in spectral_ranges]

            rebinned_lows, _, _ = binned_statistic(spectral_centers, lows, 'min', bins=rebin)
            rebinned_highs, _, _  = binned_statistic(spectral_centers, highs, 'max', bins=rebin)

            spectral_ranges = [[i, j] for i,j in zip(rebinned_lows, rebinned_highs)]
            rebin_context_tensor, _, _ = np.array([binned_statistic(spectral_centers, i, 'sum', bins=rebin)[0] for i in context_tensor])
            spectral_centers, _, _ = binned_statistic(spectral_centers, spectral_centers, 'mean', bins=rebin)
            
            print(rebin_context_tensor)
            

    warnings.warn('incorporate option for n-dim tensors for images!')
    
    for observation_ID in observation_IDs: 
        sliced_qpo_df = qpo_df.loc[qpo_df['observation_ID']==observation_ID]
        sliced_qpo_df = sliced_qpo_df.sort_values(by='frequency')
        # sliced = sliced.iloc[[np.argsort(sliced['order'])]] if order column exists, otherwise sorted by frequency? would then need to delete order column
        # keep track of reserved words 
        qpo_vector = np.array(sliced_qpo_df.drop(columns=['observation_ID'])).flatten() 

        qpo_vector = np.concatenate((qpo_vector, np.zeros(shape=max_length-len(qpo_vector))))
        qpo_tensor = np.vstack((qpo_tensor, qpo_vector))

    train_indices, test_indices = train_test_split(np.arange(0,len(qpo_tensor), 1)) # set args! 

scalar_context = ''
spectrum_context = './research and development/example_spectrum.csv'
scalar_context = './research and development/example_scalar.csv'
qpo = './research and development/example_qpo.csv'

load(qpo=qpo, 
     context=spectrum_context, 
     context_type='spectrum', rebin=1)



