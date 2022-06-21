import numpy 
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split 
np.set_printoptions(suppress=True)

def load(qpo:str, context:str, context_type:str, qpo_preprocess:dict, context_preprocess:dict, rebin:int=None, feature_range=[0.1,1]): 
    import pandas as pd 
    from collections import Counter
    from sklearn.preprocessing import OneHotEncoder   
    
    qpo_df = pd.read_csv(qpo)
    qpo_features = list(qpo_df.drop(columns=['observation_ID']))

    max_simultaneous_qpos = Counter(qpo_df['observation_ID']).most_common(1)[0][1]

    context_df = pd.read_csv(context)
    context_features = []

    observation_IDs = []

    num_qpos = []

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
            context_tensor = np.array([binned_statistic(spectral_centers, i, 'sum', bins=rebin)[0] for i in context_tensor])
            spectral_centers, _, _ = binned_statistic(spectral_centers, spectral_centers, 'mean', bins=rebin)
            
    #warnings.warn('incorporate option for n-dim tensors for images!')
    
    for observation_ID in observation_IDs: 
        sliced_qpo_df = qpo_df.loc[qpo_df['observation_ID']==observation_ID]
        sliced_qpo_df = sliced_qpo_df.sort_values(by='frequency')

        if 'order' in list(sliced_qpo_df): 
            sliced_qpo_df = sliced_qpo_df.sort_values(by='order')
            sliced_qpo_df = sliced_qpo_df.drop(columns=['order'])

        # keep track of reserved words 
        qpo_vector = np.array(sliced_qpo_df.drop(columns=['observation_ID'])).flatten() 

        qpo_vector = np.concatenate((qpo_vector, np.zeros(shape=max_length-len(qpo_vector)))) # pad qpo vector with trailing zeros if needed 
        qpo_tensor = np.vstack((qpo_tensor, qpo_vector))

    # PREPROCESS # 

    def preprocess(x:numpy.array, norm_bounds:numpy.array, encoder=None, max_simultaneous:int=None): 
    
        dim = x.ndim 
        if dim == 1: 

            min_value = norm_bounds[0]
            max_value = norm_bounds[1] 
            x = (x-min_value)/(max_value-min_value)
            x = x*(feature_range[1] - feature_range[0]) + feature_range[0]

        elif dim == 2: 

            num_features = int(len(x)/max_simultaneous_qpos)

            for i in range(max_simultaneous): 
                combined_indices = []

                for j in range(num_features): 
                    idx = i+j*num_features
                    combined_indices.append(idx)

                for idx in combined_indices: 

                    slice = x[idx]
        
                    min_value = norm_bounds[i][0]
                    max_value = norm_bounds[i][1]
                    slice = (slice-min_value)/(max_value-min_value)
                    slice = slice*(feature_range[1] - feature_range[0]) + feature_range[0]
                    x[idx]=slice

        return x 

    qpo_tensor_transposed = preprocess(np.transpose(qpo_tensor), norm_bounds=qpo_preprocess, max_simultaneous=max_simultaneous_qpos)
    qpo_tensor = np.transpose(qpo_tensor_transposed)

    context_tensor_transposed = np.transpose(context_tensor)

    for idx in range(len(context_tensor_transposed)): 
        
        slice = context_tensor_transposed[idx]
        
        context_tensor_transposed[idx] = preprocess(slice, norm_bounds=context_preprocess[idx])

    context_tensor = np.transpose(context_tensor_transposed)

    train_indices, test_indices = train_test_split(np.arange(0,len(qpo_tensor), 1, dtype=int)) # set args! 

    print(qpo_tensor)
    quit()
    print(context_tensor)

spectrum_context = './research and development/example_spectrum.csv'
scalar_context = './research and development/example_scalar.csv'
qpo = './research and development/example_qpo.csv'

load(qpo=qpo, 
     context=spectrum_context, 
     context_type='spectrum', rebin=3, qpo_preprocess=[[2,20], [0,3], [0,2]], 
     context_preprocess=[[0,100],[0,100], [0,100]])



