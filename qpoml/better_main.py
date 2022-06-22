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

        ### EVALUATE INITIALIZED  ### 
        self.all_train_indices = None # done 
        self.all_test_indices = None # done

        self.qpo_approach = None # done 

        self.evaluated_models = None # done
        self.predictions = None # done

        self.X_train = None # done
        self.X_test = None # done
        self.y_train = None # done
        self.y_test = None # done

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

                temp_one = [str(i)+'_' for i in spectral_centers]

                # lol this is an insane list comprehension 
                context_features = [temp_one[index]+str(spectral_ranges[index][0])+'-'+str(spectral_ranges[index][1]) for index in range(len(spectral_centers))]

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

    ## EVALUATE ##     
    def evaluate(self, model, model_name, evaluation_approach:str, test_proportion:float=0.1, folds:int=None) -> None: 

        self.check_loaded('evaluate')
        self.dont_do_twice('evaluate')

        from sklearn.model_selection import KFold 
        from sklearn.model_selection import train_test_split 

        random_state = self.random_state

        qpo_approach = self.qpo_approach 

        context_tensor = self.context_tensor
        qpo_tensor = self.qpo_tensor

        if qpo_approach == 'single': 

            all_train_indices = []
            all_test_indices = []

            evaluated_models = [] 
            predictions = [] 

            X_train = []
            X_test = []
            y_train = []
            y_test = []

            if evaluation_approach == 'k-fold' and folds is not None: 
                kf = KFold(n_splits=folds)
                
                train_indices = np.array([i for i, _ in kf.split(context_tensor)]).astype(int)
                test_indices = np.array([i for _, i in kf.split(context_tensor)]).astype(int)

                all_train_indices.append(train_indices)
                all_test_indices.append(test_indices)

                for train_indices_fold, test_indices_fold in zip(train_indices, test_indices): 
                    X_train_fold = context_tensor[train_indices_fold]
                    X_test_fold = context_tensor[test_indices_fold] 
                    y_train_fold = qpo_tensor[train_indices_fold]
                    y_test_fold = qpo_tensor[test_indices_fold]

                    model_fold = model 
                    model_fold.fit(X_train_fold, y_train_fold)
                    predictions.append(model_fold.predict(X_test_fold))
                    evaluated_models.append(model_fold)

                    X_train.append(X_train_fold)
                    X_test.append(X_test_fold)
                    y_train.append(y_train_fold)
                    y_test.append(y_test_fold)

            elif evaluation_approach == 'default': 

                train_indices_fold, test_indices_fold = train_test_split(np.arange(0,len(qpo_tensor), 1).astype(int), random_state=random_state)
                
                all_train_indices.append(train_indices_fold)
                all_test_indices.append(test_indices_fold)

                X_train_fold = context_tensor[train_indices_fold]
                X_test_fold = context_tensor[test_indices_fold] 
                y_train_fold = qpo_tensor[train_indices_fold]
                y_test_fold = qpo_tensor[test_indices_fold]

                model.fit(X_train_fold, y_train_fold)
                predictions.append(model.predict(X_test_fold))
                evaluated_models.append(model)

                X_train.append(X_train_fold)
                X_test.append(X_test_fold)
                y_train.append(y_train_fold)
                y_test.append(y_test_fold)

            else: 
                raise Exception('')

            self.all_train_indices = all_train_indices
            self.all_test_indices = all_test_indices

            self.evaluated_models = evaluated_models 
            self.predictions = predictions 

            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test


        elif qpo_approach == 'eurostep': 
            pass 
        
        else: 
            raise Exception('')

        

        # future idea: let users stratify by more than just internal qpo count 

        ### UPDATE ATTRIBUTES ###

        self.qpo_approach = qpo_approach

        self.evaluated = True  

    # SUPERFLUOUS # 

    # what does sandman mean in the old slang? e.g. in hushaby song 

    ## UTILITIES ## 

    def performance_statistics(self): 
        self.check_loaded_evaluated('performance_statistics')

        from sklearn.metrics import mean_absolute_error, mean_squared_error

        statistics = {}

        if self.qpo_approach=='single': 
            predictions = self.predictions
            y_test = self.y_test 

            mae = None 
            mse = None

            if len(predictions)==1: 
                predictions = predictions[0].flatten()
                y_test = y_test[0].flatten()
                mse = mean_squared_error(y_test, predictions)
                mae = mean_absolute_error(y_test, predictions)

            else: 
                
                mse = []
                mae = []

                for prediction, true in zip(predictions, y_test): 
                    mse.append(mean_squared_error(true, prediction))
                    mae.append(mean_absolute_error(true, prediction))
                
            statistics['mse'] = mse
            statistics['mae'] = mae

        return statistics  

    ## UTILITY WRAPPERS ## 

    ### POST LOAD ### 

    def correlation_matrix(self): 
        self.check_loaded('correlation_matrix')
        from qpoml.utilities import correlation_matrix 

        data = self.context_tensor
        columns = self.context_features

        data = pd.DataFrame(data, columns=columns)

        corr, cols = correlation_matrix(data=data)

        return corr, cols

    def dendrogram(self):
        self.check_loaded('dendrogram')
        from qpoml.utilities import dendrogram 

        data = self.context_tensor
        columns = self.context_features

        data = pd.DataFrame(data, columns=columns) 

        corr, dist_linkage, cols = dendrogram(data=data)
        
        return corr, dist_linkage, cols 

    def calculate_vif(self):
        self.check_loaded('calculate_vif')
        from qpoml.utilities import calculate_vif 

        data = self.context_tensor 
        columns = self.context_features 

        data = pd.DataFrame(data, columns=columns)

        vif_df = calculate_vif(data=data)

        return vif_df 

    ### POST EVALUATION ### 

    def results_regression(self, which:list, fold:int=None):
        from qpoml.utilities import results_regression
        self.check_evaluated('feature_importances') 

        model = self.evaluated_models
        predictions = self.predictions 
        y_test = self.y_test 

        if self.qpo_approach == 'k-fold' and fold is not None: 
            model = model[fold]
            predictions = predictions[fold]
            y_test = y_test[fold] 

        else: 
            model = model[0]
            predictions = predictions[0]
            y_test = y_test[0] 

        regression_x, regression_y, mb, stats = results_regression(y_test=y_test, predictions=predictions, which=which)

        return regression_x, regression_y, mb, stats

    def feature_importances(self, feature_names:list, kind:str='kernel-shap', fold:int=None):
        r'''
        fold is if the user previously chose to do k-fold cross validation, 0 index of models to select feature importances from 
        '''
        from qpoml.utilities import feature_importances
        self.check_evaluated('feature_importances') 

        model = self.evaluated_models
        X_test = self.X_test 
        y_test = self.y_test 

        if self.qpo_approach == 'k-fold' and fold is not None: 
            model = model[fold]
            X_test = X_test[fold]
            y_test = y_test[fold] 

        else: 
            model = model[0]
            X_test = X_test[0]
            y_test = y_test[0]

        feature_importances_arr, feature_names, importances_df = feature_importances(model, X_test, y_test, feature_names=feature_names, kind=kind)
        
        return feature_importances_arr, feature_names, importances_df

    ## PLOTTING WRAPPERS ## 

    ### POST LOAD ### 

    def plot_correlation_matrix(self, ax=None, matrix_style:str='default'):
        self.check_loaded('plot_correlation_matrix')
        from qpoml.plotting import plot_correlation_matrix

        data = self.context_tensor 
        columns = self.context_features 

        data = pd.DataFrame(data, columns=columns)

        plot_correlation_matrix(data=data, ax=ax, matrix_style=matrix_style)

    def plot_pairplot(self, steps=False, ax=None): 
        self.check_loaded('plot_pairplot')
        from qpoml.plotting import plot_pairplot 

        data = self.context_tensor 
        columns = self.context_features 

        data = pd.DataFrame(data, columns=columns)

        plot_pairplot(data=data, steps=steps, ax=ax)

    def plot_dendrogram(self, ax=None): 
        self.check_loaded('plot_dendrogram')
        from qpoml.plotting import plot_dendrogram 

        data = self.context_tensor
        columns = self.context_features

        data = pd.DataFrame(data, columns=columns) 

        plot_dendrogram(data=data, ax=ax)

    def plot_vif(self, cutoff:int=10, ax=None): 
        self.check_loaded('plot_vif')
        from qpoml.plotting import plot_vif 

        data = self.calculate_vif() 

        plot_vif(data=data, cutoff=cutoff, ax=ax)

    ### POST EVALUATION ### 

    def plot_results_regression(self, feature_name:str, which:list, kind:str='kernel-shap', ax=None, xlim:list=[0.1,1], fold:int=None):
        self.check_evaluated('plot_results_regression')
        from qpoml.plotting import plot_results_regression

        model = self.evaluated_models 
        X_test = self.X_test 
        y_test = self.y_test 
        predictions = self.predictions 

        if self.qpo_approach == 'k-fold' and fold is not None: 
            model = model[fold]
            predictions = predictions[fold]
            X_test = X_test[fold]
            y_test = y_test[fold] 

        else: 
            model = model[0]
            predictions = predictions[0]
            X_test = X_test[0]
            y_test = y_test[0] 

        plot_results_regression(model=model, X_test=X_test, y_test=y_test, predictions=predictions, feature_name=feature_name, which=which, kind=kind, ax=ax, xlim=xlim)
 
    def plot_feature_importances(self, kind:str='kernel-shap', ax=None, fold:int=None):
        self.check_evaluated('plot_feature_importances')
        from qpoml.plotting import plot_feature_importances
        
        model = self.evaluated_models 
        X_test = self.X_test 
        y_test = self.y_test 
        predictions = self.predictions 

        if self.qpo_approach == 'k-fold' and fold is not None: 
            model = model[fold]
            predictions = predictions[fold]
            X_test = X_test[fold]
            y_test = y_test[fold] 

        else: 
            model = model[0]
            predictions = predictions[0]
            X_test = X_test[0]
            y_test = y_test[0] 

        feature_names = self.context_features

        plot_feature_importances(model=model, X_test=X_test, y_test=y_test, feature_names=feature_names, kind=kind, ax=ax)

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