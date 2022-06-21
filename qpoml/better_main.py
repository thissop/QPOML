import numpy 
import pandas
import warnings 
import numpy as np
import pandas as pd 

class collection: 

    qpo_reserved_words = ['observation_ID', 'order']

    # REQUIRED # 

    ## INITIALIZE ##
    def __init__(self, random_state:int=42) -> None:
        
        ### global ###

        self.random_state = random_state # apply to all!

        self.loaded = False # accounted for 
        self.evaluated = False # accounted for 

        self.observation_IDs = None # done 

        self.context_is_spectrum = False # done 
        self.context_tensor = None # done 
        self.context_features = None # done 
        self.spectral_ranges = None # done  
        self.spectral_centers = None # done 
        
        self.qpo_tensor = None # done 
        self.num_qpos = None # done 
        self.max_simultaneous_qpos = None # done 
        self.qpo_features = None # done 
        self.qpo_approach = None # done 

        self.train_test_indices = None 

        ### stage tracking ### 
        
    ## LOAD ## 
    def load(self, qpo_csv:str, context_csv:str, context_type:str, context_preprocess, qpo_preprocess:dict, qpo_approach:str='single', spectrum_approach:str='by-row', rebin:int=None) -> None: 
        
        from collections import Counter
        from utilities import preprocess1d

        # check so you can do it once it's already been loaded or evaluated 

        ### CONTEXT ### 
        # can context be loaded independently? 

        context_df = pd.read_csv(context_csv)
        temp_df = context_df.drop(columns=['observation_ID'])
        
        observation_IDs = list(context_df['observation_ID'])
        context_features = list(temp_df)
        context_tensor = np.array(temp_df)

        if context_type=='spectrum': 
            self.context_is_spectrum = True 
            
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

            ### NORMALIZE CONTEXT SPECTRUM ### 
            temp_tensor = None 
            if spectrum_approach == 'by-row' or spectrum_approach == 'as-is': 
                temp_tensor = context_tensor 
            elif spectrum_approach == 'by-column': 
                temp_tensor = np.transpose(context_tensor)
            else: 
                raise Exception('')
            for index, arr in enumerate(temp_tensor): 
                    arr = preprocess1d(arr, context_preprocess)
                    temp_tensor[index] = arr 
            if spectrum_approach == 'by-column': 
                context_tensor = np.transpose(temp_tensor)
            else: 
                context_tensor = temp_tensor 

            self.spectral_ranges = spectral_ranges 
            self.spectral_centers = spectral_centers 

        else: 
            for index, arr in enumerate(context_tensor): 
                arr = preprocess1d(arr, context_preprocess[context_features[index]])
                context_tensor[index] = arr 

        ### QPO ### 

        # bruhhhhhhh...QPO features should be normalized as the columns in the dataframe, and then they should be passed to the tensor. 
        # normalize them in 0.1-1.0 range, and then pad single approach with zeros after. 
        # when I am evaluating the results afterwards, will need to pad eurostep vectors with zeros so that the results can be the same for each

        num_qpos = []

        qpo_df = pd.read_csv(qpo_csv)
        qpo_features = list(qpo_df.drop(columns=self.qpo_reserved_words))

        max_simultaneous_qpos = Counter(qpo_df['observation_ID']).most_common(1)[0][1]
        max_length = max_simultaneous_qpos*len(qpo_features)
        qpo_tensor = []

        for qpo_feature in qpo_features: 
            if qpo_feature not in self.qpo_reserved_words: # reserved QPO words 
                qpo_df[qpo_feature] = preprocess1d(x=qpo_df[qpo_feature], preprocess=qpo_preprocess[qpo_feature])

        for observation_ID in observation_IDs: 
            sliced_qpo_df = qpo_df.loc[qpo_df['observation_ID']==observation_ID]
            sliced_qpo_df = sliced_qpo_df.sort_values(by='frequency')

            num_qpos.append(len(sliced_qpo_df.index))

            if 'order' in list(sliced_qpo_df): 
                sliced_qpo_df = sliced_qpo_df.sort_values(by='order')
                sliced_qpo_df = sliced_qpo_df.drop(columns=['order'])

            # keep track of reserved words 
            qpo_vector = np.array(sliced_qpo_df.drop(columns=['observation_ID'])).flatten() 

            qpo_tensor.append(qpo_vector)

    
        if qpo_approach == 'single': 
            # pad with zeros  
            for index, arr in enumerate(qpo_tensor): 
                arr = np.concatenate((arr, np.zeros(shape=max_length-len(arr)))) 
                qpo_tensor[index] = arr 

        qpo_tensor = np.array(qpo_tensor)

        ### UPDATE ATTRIBUTES ### 
        self.observation_IDs = observation_IDs 
        self.num_qpos = num_qpos 
        self.context_tensor = context_tensor 
        self.qpo_tensor = qpo_tensor 
        self.max_simultaneous_qpos = max_simultaneous_qpos
        self.qpo_approach = qpo_approach
        self.context_features = context_features 
        self.qpo_features = qpo_features 

        self.loaded = True 

        print(context_tensor)
        print(qpo_tensor)

    ## EVALUATE ##     
    def evaluate(self): 

        # check so you can't do it again once it's already been evaluated 





        ### UPDATE ATTRIBUTES ###

        self.evaluated = True  

    # SUPERFLUOUS # 

    ## UTILITY WRAPPERS ## 

    ## PLOTTING WRAPPERS ## 