import numpy 
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split 
np.set_printoptions(suppress=True)

def load(qpo:str, context:str, context_type:str, qpo_preprocess:dict, context_preprocess:dict, rebin:int=None): 
    import pandas as pd 
    from collections import Counter
    from sklearn.preprocessing import OneHotEncoder   
    
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
            context_tensor = np.array([binned_statistic(spectral_centers, i, 'sum', bins=rebin)[0] for i in context_tensor])
            spectral_centers, _, _ = binned_statistic(spectral_centers, spectral_centers, 'mean', bins=rebin)
            
    #warnings.warn('incorporate option for n-dim tensors for images!')
    
    for observation_ID in observation_IDs: 
        sliced_qpo_df = qpo_df.loc[qpo_df['observation_ID']==observation_ID]
        sliced_qpo_df = sliced_qpo_df.sort_values(by='frequency')
        # sliced = sliced.iloc[[np.argsort(sliced['order'])]] if order column exists, otherwise sorted by frequency? would then need to delete order column
        # keep track of reserved words 
        qpo_vector = np.array(sliced_qpo_df.drop(columns=['observation_ID'])).flatten() 

        qpo_vector = np.concatenate((qpo_vector, np.zeros(shape=max_length-len(qpo_vector)))) # pad qpo vector with trailing zeros if needed 
        qpo_tensor = np.vstack((qpo_tensor, qpo_vector))

    # PREPROCESS # 

    def preprocess(x:numpy.array, method, encoder=None, max_simultaneous:int=None): 
    
        dim = x.ndim 
        if dim == 1: 

            encoded_key = None

            if type(method) is str: 

                if method=='normalize': 
                    min_value = np.min(x)
                    max_value = np.max(x)
                    x = (x-min_value)/(max_value-min_value), (min_value, max_value)

                elif method == 'standardize':
                    mean = np.mean(x)
                    sigma = np.std(x)  
                    x = (x-mean)/sigma

                elif method == 'categorical': 

                    x = x.reshape(-1,1)
        
                    original_classes = encoder.categories_[0]
                    encoded_classes = encoder.transform(original_classes.reshape(-1,1)).toarray()
                    encoded_key = {'original_classes':original_classes, 'encoded_classes':encoded_classes}
                    x = encoder.transform(x).toarray()
                    print(x)
                    x = np.array([arr for arr in x]) 
                    print(x)

                elif method == 'as-is':
                    x = x

            elif type(method) is list:

                min_value = method[0]
                max_value = method[1] 
                x = (x-min_value)/(max_value-min_value), (min_value, max_value)

        elif dim == 2: 

            num_features = len(x)

            for i in range(max_simultaneous+1): 
                combined_indices = []

                for j in range(0, num_features+1, num_features): 
                    idx = i+j
                    combined_indices.append(idx)

                flat = x[combined_indices].flatten()

                slice_method = method 
                if type(method) is dict: 
                    slice_method = method.values[i] # preprocess dictionary items need to be in same order as the values appear in csv!

                for idx in combined_indices:
                    slice = x[idx] 
                    x[idx] = preprocess(x=slice, method=slice_method)

        if type(method) is str and method=='categorical':
            return x, encoded_key 
        
        else: 
            return x 

    qpo_tensor_transposed = preprocess(np.transpose(qpo_tensor), method=qpo_preprocess, max_simultaneous=max_simultaneous_qpos)
    qpo_tensor = np.transpose(qpo_tensor_transposed)

    context_tensor_transposed = np.transpose(context_tensor)

    for idx in range(len(context_tensor_transposed)): 
        
        slice = context_tensor_transposed[idx]
        method = context_preprocess.values()[idx]

        if type(method) is str and method == 'categorical': 
            encoder = OneHotEncoder()
            encoder.fit(slice)

            context_tensor_transposed[idx] = preprocess(slice, method=method, encoder=encoder)

        else: 
            context_tensor_transposed[idx] = preprocess(slice, method=method)

    context_tensor = np.transpose(context_tensor_transposed)

    train_indices, test_indices = train_test_split(np.arange(0,len(qpo_tensor), 1, dtype=int)) # set args! 

    print(qpo_tensor)
    print(context_tensor)

spectrum_context = './research and development/example_spectrum.csv'
scalar_context = './research and development/example_scalar.csv'
qpo = './research and development/example_qpo.csv'

load(qpo=qpo, 
     context=spectrum_context, 
     context_type='spectrum', rebin=2, qpo_preprocess={'frequency':[1,20], 'width':[0,3], 'amplitude':[0,2]}, 
     context_preprocess={'first':'normalize', 'second':'normalize'})



