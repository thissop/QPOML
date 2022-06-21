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
        
        self.dont_do_twice('load')

        from collections import Counter
        from qpoml.utilities import preprocess1d

        # check so you can do it once it's already been loaded or evaluated 

        ### CONTEXT ### 

        context_df = pd.read_csv(context_csv).sample(frac=1, random_state=self.random_state) # shuffle 
        temp_df = context_df.drop(columns=['observation_ID'])
        
        observation_IDs = list(context_df['observation_ID'])
        context_features = list(temp_df)
        context_tensor = np.array(temp_df)

        if context_type=='spectrum': 
            self.context_is_spectrum = True 

            context_tensor = context_tensor.astype(float)
            
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
            transposed = np.transpose(context_tensor)
            for index, arr in enumerate(transposed): 
                arr = preprocess1d(arr, context_preprocess[context_features[index]])
                transposed[index] = arr 
            
            context_tensor = np.transpose(transposed)

        ### QPO ### 

        num_qpos = []

        qpo_df = pd.read_csv(qpo_csv)
        qpo_features = [i for i in list(qpo_df) if i not in self.qpo_reserved_words]

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

        qpo_tensor = np.array(qpo_tensor, dtype=object)

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

        print(qpo_tensor)

    ## EVALUATE ##     
    def evaluate(self, model, model_name, evaluation_approach:str, test_proportion:float=0.1, folds:int=None): 

        self.check_loaded('evaluate')
        self.dont_do_twice('evaluate')

        from sklearn.model_selection import KFold 
        from sklearn.model_selection import train_test_split 

        random_state = self.random_state

        qpo_approach = self.qpo_approach 

        context_tensor = self.context_tensor
        qpo_tensor = self.qpo_tensor

        if qpo_approach == 'single': 

            if evaluation_approach == 'k-fold' and folds is not None: 
                kf = KFold(n_splits=folds, random_state=random_state)
                
                train_indices = np.array([i for i, _ in kf.split(context_tensor, random_state=random_state)]).astype(int)
                test_indices = np.array([i for _, i in kf.split(context_tensor, random_state=random_state)]).astype(int)

                evaluated_models = [] 
                predictions = [] 

                global_X_train = []
                global_X_test = []
                global_y_train = []
                global_y_test = []

                for train_indices_fold, test_indices_fold in zip(train_indices, test_indices): 
                    X_train = context_tensor[train_indices_fold]
                    X_test = context_tensor[train_indices_fold] 
                    y_train = qpo_tensor[train_indices_fold]
                    y_test = qpo_tensor[test_indices_fold]

                    model.fit(X_train, y_train)
                    predictions.append(model.predict(X_test, y_test))
                    evaluated_models.append(model)

                    global_X_train.append(X_train)
                    global_X_test.append(X_test)
                    global_y_train.append(y_train)
                    global_y_test.append(y_test)

            elif evaluation_approach == 'default': 

                train_indices, test_indices = train_test_split(np.arange(0,len(qpo_tensor), 1).astype(int), random_state=random_state)
                



            else: 
                raise Exception('')


        elif qpo_approach == 'eurostep': 
            pass 
        else: 
            raise Exception('')

        

        # future idea: let users stratify by more than just internal qpo count 

        ### UPDATE ATTRIBUTES ###

        self.evaluated = True  

    # SUPERFLUOUS # 

    ## UTILITY WRAPPERS ## 

    def performance_statistics(self): 
        self.check_loaded_evaluated('performance_statistics')

    ## PLOTTING WRAPPERS ## 

    ## GOTCHAS ## 
    def check_loaded(self, function:str):
        if not self.loaded: 
            raise Exception('collection must be loaded before '+function+'() function can be accessed') 
    
    def check_evaluated(self, function:str):
        if not self.evaluated: 
            raise Exception('collection must be evaluated before '+function+'() function can be accessed') 

    def check_loaded_evaluated(self, function:str): 
        self.check_loaded(function)
        self.check_evaluated(function)

    def dont_do_twice(self, function:str): 
        if function=='load' and self.loaded: 
            raise Exception('the function '+function+'() has already been executed and this step cannot be repeated on the object now')
        elif function=='evaluate' and self.evaluated: 
            raise Exception('the function '+function+'() has already been executed and this step cannot be repeated on the object now')  