import os 
import numpy
import pandas
import warnings
import numpy as np
import pandas as pd

np.set_printoptions(suppress=True)

wh1 = False
if os.path.exists('/ar1/PROJ/fjuhsd/personal/thaddaeus/github/QPOML/qpoml/stylish.mplstyle'):
    wh1 = True 

class collection:

    qpo_reserved_words = ["observation_ID", "order"]

    # REQUIRED #

    ## INITIALIZE ##

    def __init__(self, random_state: int = 42) -> None:

        ### global ###

        self.random_state = random_state  # apply to all!

        self.loaded = False  # accounted for
        self.evaluated = False  # accounted for

        self.observation_IDs = None  # done

        self.units = None # done 

        self.context_is_spectrum = False  # done
        self.context_tensor = None  # done
        self.context_features = None  # done
        self.spectral_ranges = None  # done
        self.spectral_centers = None  # done

        self.qpo_tensor = None  # done
        self.num_qpos = None  # done
        self.max_simultaneous_qpos = None  # done
        self.qpo_features = None  # done

        self.qpo_preprocess1d_tuples = None # done

        self.classification_or_regression = None # done 

        ### EVALUATE INITIALIZED  ###
        self.train_indices = None  # done
        self.test_indices = None  # done

        self.qpo_preprocess = None 
        self.context_preprocess = None # not really needed so not implemented yet 

        self.X_train = None  # done
        self.X_test = None  # done
        self.y_train = None  # done
        self.y_test = None  # done

        self.train_observation_IDs = None # done
        self.test_observation_IDs = None # done

        # Gridsearch Related
        self.gridsearch_scores = None 
        self.gridsearch_stds = None 
        self.gridsearch_params = None 
        self.best_params = None 

        self.evaluated_model = None  # done
        self.predictions = None  # done

        self.TPRs = None # done 
        self.FPRs = None # done 
        self.auc_scores = None # done 

        self.fold_performances = None # done

    ## LOAD ##

    def load(
        self,
        qpo_csv: str,
        context_csv: str,
        context_preprocess,
        approach:str,
        qpo_preprocess=None,
        units:dict=None) -> None:

        r"""
        _Class method for loading collection object_
        Parameters
        ----------
        qpo_csv : `str`
            File path to correctly formatted csv file with QPO information
        context_csv : `str`
            File path to correctly formatted csv file with context information
        context_preprocess : `dict` or `str`
            Fix this
        approach : str
            Either "regression" or "classification" 
        qpo_preprocess : `dict` or `str`
            Fix this. If none but approach is regression, then all features will be "locally" min-max normalized. 
        Returns
        -------
        """

        self.dont_do_twice("load")

        from collections import Counter
        from qpoml.utilities import preprocess1d

        # check so you can do it once it's already been loaded or evaluated

        ### CONTEXT ###

        context_df = pd.read_csv(context_csv).sample(
            frac=1, random_state=self.random_state
        )  # shuffle
        temp_df = context_df.drop(columns=["observation_ID"])

        observation_IDs = list(context_df["observation_ID"])
        context_features = list(temp_df)
        context_tensor = np.array(temp_df)

        transposed = np.transpose(context_tensor)
        for index, arr in enumerate(transposed):
            arr, (_, _, _) = preprocess1d(arr, context_preprocess[context_features[index]])
            transposed[index] = arr

        context_tensor = np.transpose(transposed)

        ### QPO ###

        qpo_df = pd.read_csv(qpo_csv)
        qpo_tensor = []
        
        if approach == 'regression':
            num_qpos = []
            qpo_features = [i for i in list(qpo_df) if i not in self.qpo_reserved_words]

            max_simultaneous_qpos = Counter(qpo_df["observation_ID"]).most_common(1)[0][1]
            max_length = max_simultaneous_qpos * len(qpo_features)

            qpo_preprocess1d_tuples = {}
            for qpo_feature in qpo_features:
                if qpo_feature not in self.qpo_reserved_words:  # reserved QPO words
                    
                    preprocess_method = None 

                    if qpo_preprocess is None: 
                        preprocess_method = 'normalize'
                    else: 
                        preprocess_method = qpo_preprocess[qpo_feature]

                    modified, preprocess1d_tuple = preprocess1d(x=qpo_df[qpo_feature], preprocess=preprocess_method)

                    qpo_df[qpo_feature] = modified

                    qpo_preprocess1d_tuples[qpo_feature] = preprocess1d_tuple

            self.qpo_preprocess1d_tuples = qpo_preprocess1d_tuples

            for observation_ID in observation_IDs:
                sliced_qpo_df = qpo_df.loc[qpo_df["observation_ID"] == observation_ID]
                sliced_qpo_df = sliced_qpo_df.sort_values(by="frequency")

                num_qpos.append(len(sliced_qpo_df.index))

                if "order" in list(sliced_qpo_df):
                    sliced_qpo_df = sliced_qpo_df.sort_values(by="order")
                    sliced_qpo_df = sliced_qpo_df.drop(columns=["order"])

                # keep track of reserved words
                qpo_vector = np.array(
                    sliced_qpo_df.drop(columns=["observation_ID"])
                ).flatten()

                qpo_tensor.append(qpo_vector)

            for index, arr in enumerate(qpo_tensor):
                arr = np.concatenate((arr, np.zeros(shape=max_length - len(arr))))
                qpo_tensor[index] = arr

            qpo_tensor = np.array(qpo_tensor, dtype=object)
            self.qpo_features = qpo_features
            self.max_simultaneous_qpos = max_simultaneous_qpos
            self.num_qpos = num_qpos

        else: 
            qpo_columns = np.array(list(qpo_df))
            
            qpo_df = context_df.merge(qpo_df, on='observation_ID')

            qpo_df = qpo_df[qpo_columns]

            class_column_name = qpo_columns[qpo_columns!='observation_ID'][0]

            qpo_tensor = np.array(qpo_df[class_column_name]) # ravel/reshape this? 

        ### UPDATE ATTRIBUTES ###
        self.observation_IDs = observation_IDs
        
        self.context_tensor = context_tensor
        self.context_features = context_features
        self.qpo_tensor = qpo_tensor

        self.units = units

        self.loaded = True
        self.classification_or_regression = approach 

    ## EVALUATE ##

    def evaluate(
        self,
        model,
        test_proportion: float = 0.1,
        folds: int = None,
        repetitions: int = None, 
        gridsearch_dictionary:dict=None,
        stratify=None, 
        multiclass:bool=False) -> None:
        
        r"""
        _Evaluate an already initiated and loaded model_
        Parameters
        ----------
        model : `object`
            An initialized regressor object from another class, e.g. sklearn. It will be cloned and then have parameters reset, so it's okay (it's actually nesessary) that it is initalized
        test_proportion : `float`
            Default is `0.1`; proportion of values to reserve for test set
        folds : `int`
            Default is `None`; if set to some integer, the model will be validated via K-Fold validation, with `K=folds`
        stratify : bool or dict
            if true and classification, classes are used for stratification. If a separate iterable object, that object is used for stratification (e.g. if you make classes that correspond by index to context tensor that are qpo class and object number, for multiple sources)
            if True (default) stratifies splitting on class output vector. If it is a dictionary, then the dictionary needs to have keys as observations and the corresponding items as the values upon which they will be fed and stratified on. can only be boolean if clasification. otherwise, needs to be dictionary. dictionary works for reg or class tho. needs to have 'observation_ID':[] and 'class':[] for stratification
       
        
        Returns
        -------
        To-Do 
        -----
        - fix classification_or_regression to easier format (e.g. boolean) 
        Notes 
        -----
        - I know I already said this elsewhere, but stratify can only be bool for type(load approach) == bool. Otherwise, especially for regression, it needs to be a dictionary of form {'observation_ID':[], 'class':[]}
        - Default does not stratify! only k-fold or repeat k-fold! 

        - hyperparameter_dictionary gets fed in to the model for actual testing, but gridsearch_dictionary is a dictionary of lists that are used as parameter combinations for gridsearch

        """

        self.check_loaded("evaluate")
        self.dont_do_twice("evaluate")

        import sklearn 
        from sklearn.model_selection import train_test_split
        from qpoml.utilities import gridsearch

        random_state = self.random_state
        context_tensor = self.context_tensor
        qpo_tensor = self.qpo_tensor
        observation_IDs = np.array(self.observation_IDs)
        classification_or_regression = self.classification_or_regression

        indices = list(range(len(self.observation_IDs)))

        # NEED TO SPLIT HERE STRATIFIED!!

        if stratify is not None: 
            if type(stratify) is bool: 
                if stratify: 
                    if classification_or_regression == 'classification':
                        train_indices, test_indices, train_observation_IDs, test_observation_IDs, context_tensor_train, context_tensor_test, qpo_tensor_train, qpo_tensor_test = train_test_split(indices, observation_IDs, context_tensor, qpo_tensor, test_size=test_proportion, random_state=random_state, stratify=qpo_tensor)
                    else: 
                        qpos_per_obs = []
                        for i in qpo_tensor: 
                            num = int(len(np.where(i!=0.1)[0])/len(self.qpo_features))
                            qpos_per_obs.append(num)
                        train_indices, test_indices, train_observation_IDs, test_observation_IDs, context_tensor_train, context_tensor_test, qpo_tensor_train, qpo_tensor_test = train_test_split(indices, observation_IDs, context_tensor, qpo_tensor, test_size=test_proportion, random_state=random_state, stratify=qpos_per_obs)
                else: 
                    train_indices, test_indices, train_observation_IDs, test_observation_IDs, context_tensor_train, context_tensor_test, qpo_tensor_train, qpo_tensor_test = train_test_split(indices, observation_IDs, context_tensor, qpo_tensor, test_size=test_proportion, random_state=random_state)
            else: 
                train_indices, test_indices, train_observation_IDs, test_observation_IDs, context_tensor_train, context_tensor_test, qpo_tensor_train, qpo_tensor_test = train_test_split(indices, observation_IDs, context_tensor, qpo_tensor, test_size=test_proportion, random_state=random_state, stratify=stratify)
        else: 
            train_indices, test_indices, train_observation_IDs, test_observation_IDs, context_tensor_train, context_tensor_test, qpo_tensor_train, qpo_tensor_test = train_test_split(indices, observation_IDs, context_tensor, qpo_tensor, test_size=test_proportion, random_state=random_state)
 
        # GRID SEARCH --> MAKE INTERNAL, FIX PIPELINE!

        if self.classification_or_regression == 'classification':
            (_, best_params), (scores, stds, params, best_model_fold_scores), (FPRs, TPRs, auc_scores) = gridsearch(model=model, observation_IDs=train_observation_IDs, X=context_tensor_train, y=qpo_tensor_train, gridsearch_dictionary=gridsearch_dictionary, class_or_reg=classification_or_regression, stratify=stratify, folds=folds, repetitions=repetitions, random_state=random_state, num_qpo_features=None, multiclass=multiclass) # making this default! 

        else: 
            (_, best_params), (scores, stds, params, best_model_fold_scores), (FPRs, TPRs, auc_scores) = gridsearch(model=model, observation_IDs=train_observation_IDs, X=context_tensor_train, y=qpo_tensor_train, gridsearch_dictionary=gridsearch_dictionary, class_or_reg=classification_or_regression, stratify=stratify, folds=folds, repetitions=repetitions, random_state=random_state, num_qpo_features=len(self.qpo_features)) # making this default! 

        local_model = sklearn.base.clone(model)

        local_model = local_model.set_params(**best_params)

        local_model.fit(context_tensor_train, qpo_tensor_train)
        predictions = local_model.predict(context_tensor_test)

        if classification_or_regression == 'classification': 
            predictions = np.array(predictions).flatten()
        
        # UPDATE INTERNAL PARAMETERS

        self.train_indices = train_indices # done
        self.test_indices = test_indices # done

        self.evaluated_model = local_model # done
        self.predictions = predictions # done

        self.X_train = context_tensor_train # done
        self.X_test = context_tensor_test # done
        self.y_train = qpo_tensor_train # done
        self.y_test = qpo_tensor_test # done
        self.train_observation_IDs = train_observation_IDs # done
        self.test_observation_IDs = test_observation_IDs # done

        self.gridsearch_scores = scores # done 
        self.gridsearch_stds = stds # done
        self.gridsearch_params = params # done
        self.best_params = best_params # done

        self.FPRs = FPRs
        self.TPRs = TPRs
        self.auc_scores = auc_scores

        self.fold_performances = best_model_fold_scores

        ### UPDATE ATTRIBUTES ###

        self.evaluated = True
    
    # SUPERFLUOUS #

    # what does sandman mean in the old slang? e.g. in hushaby song

    ## UTILITIES ##

    def get_performance_statistics(self, predicted_feature_name:str=None):
        r"""

        FIX THIS!!! DOESN'T WORK/WILL 99% NOT WORK AFTER CURRENT CHANGES!!
        

        _Return model performance statistics_
        Parameters
        ----------
        Returns
        -------
        statistics : `dict`
            Dictionary of performance statistics. Currently contains `mae` and `mse`
        predicted_feature_name : str
            If not None, this feature name will be used to undo the preproccessing on the vector. 
        """

        self.check_loaded_evaluated("performance_statistics")

        from sklearn.metrics import mean_absolute_error, mean_squared_error
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from qpoml.utilities import unprocess1d 

        classification_or_regression = self.classification_or_regression

        statistics = {}

        predictions = self.predictions
        y_test = self.y_test

        if len(predictions) == 1:
            predictions = predictions[0].flatten()
            y_test = y_test[0].flatten()

            if classification_or_regression == 'classification':
        
                statistics['accuracy'] = [accuracy_score(y_test, predictions)]
                statistics['precision'] = [precision_score(y_test, predictions)]
                statistics['recall'] = [recall_score(y_test, predictions)]
                statistics['f1'] = [f1_score(y_test, predictions)]

            else: 
                
                statistics["mse"] = [mean_squared_error(y_test, predictions)]
                statistics["mae"] = [mean_absolute_error(y_test, predictions)]

        else:
            
            if classification_or_regression == 'classification':

                accuracy = []
                precision = []
                recall = []
                f1 = []

                for true, prediction in zip(y_test, predictions):
                    accuracy.append(accuracy_score(true, prediction))
                    precision.append(precision_score(true, prediction))
                    recall.append(recall_score(true, prediction))
                    f1.append(f1_score(true, prediction))

                statistics['accuracy'] = accuracy 
                statistics['precision'] = precision
                statistics['recall'] = recall
                statistics['f1'] = f1 

            else: 

                mse = []
                mae = []

                for true, prediction in zip(y_test, predictions):

                    if predicted_feature_name is not None: 
                        preprocess1d_tuple=self.qpo_preprocess1d_tuples[predicted_feature_name]
                        true = unprocess1d(true, preprocess1d_tuple)
                        prediction = unprocess1d(prediction, preprocess1d_tuple)

                    mse.append(mean_squared_error(true, prediction))
                    mae.append(mean_absolute_error(true, prediction))

                statistics["mse"] = mse
                statistics["mae"] = mae

        return statistics
    
    ## UTILITY WRAPPERS ##
    
    ### POST LOAD ###

    def correlation_matrix(self):
        r"""
        _Class wrapper to `utils.correlation_matrix`_
        Parameters
        ----------
        Returns
        -------
        fix after adding to utils correlation matrix docs!
        """

        self.check_loaded("correlation_matrix")
        from qpoml.utilities import correlation_matrix

        data = self.context_tensor
        columns = self.context_features

        data = pd.DataFrame(data, columns=columns)

        corr, cols = correlation_matrix(data=data)

        return corr, cols

    def dendrogram(self):
        self.check_loaded("dendrogram")
        from qpoml.utilities import dendrogram

        data = self.context_tensor
        columns = self.context_features

        data = pd.DataFrame(data, columns=columns)

        corr, dist_linkage, cols = dendrogram(data=data)

        return corr, dist_linkage, cols

    def calculate_vif(self):
        self.check_loaded("calculate_vif")
        from qpoml.utilities import calculate_vif

        data = self.context_tensor
        columns = self.context_features

        data = pd.DataFrame(data, columns=columns)

        vif_df = calculate_vif(data=data)

        return vif_df

    ### POST EVALUATION ###

    def results_regression(self, feature_name:str, which: list):

        r'''
        
        Arguments
        ---------
        feature_name : str
            Since preprocessing is handled under the hood within the collection class, users won't have access to their preprocess1d tuples. Thus, they need to also give the name of the feature (e.g. 'frequency') so this method can locate a saved copy of the preprocess1d tuple that matches with an output QPO feature that was registered when the collection object was loaded.
        '''


        from qpoml.utilities import results_regression

        self.check_evaluated("feature_importances")

        predictions = self.predictions
        y_test = self.y_test

        regression_x, regression_y, linregress_result = results_regression(y_test=y_test, predictions=predictions, which=which, 
                                                                   preprocess1d_tuple=self.qpo_preprocess1d_tuples[feature_name])

        return regression_x, regression_y, linregress_result

    def feature_importances(self, feature_names: list, kind: str = "kernel-shap"):
        r"""
        fold is if the user previously chose to do k-fold cross validation, 0 index of models to select feature importances from
        """

        from qpoml.utilities import feature_importances

        self.check_evaluated("feature_importances")

        model = self.evaluated_model
        X_test = self.X_test
        y_test = self.y_test

        mean_importances_df, importances_df = feature_importances(model=model, X_test=X_test, y_test=y_test, feature_names=feature_names, kind=kind)

        return mean_importances_df, importances_df

    def get_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

    ## PLOTTING WRAPPERS ##

    ### POST LOAD ###

    def plot_correlation_matrix(self, ax=None, matrix_style: str = "default"):
        self.check_loaded("plot_correlation_matrix")
        from qpoml.plotting import plot_correlation_matrix

        data = self.context_tensor
        columns = self.context_features

        data = pd.DataFrame(data, columns=columns)

        plot_correlation_matrix(data=data, ax=ax, matrix_style=matrix_style)

    def plot_pairplot(self, steps=False, ax=None):
        self.check_loaded("plot_pairplot")
        from qpoml.plotting import plot_pairplot

        data = self.context_tensor
        columns = self.context_features

        data = pd.DataFrame(data, columns=columns)

        plot_pairplot(data=data, steps=steps, ax=ax)

    def plot_dendrogram(self, ax=None):
        self.check_loaded("plot_dendrogram")
        from qpoml.plotting import plot_dendrogram

        data = self.context_tensor
        columns = self.context_features

        data = pd.DataFrame(data, columns=columns)

        plot_dendrogram(data=data, ax=ax)

    def plot_vif(self, cutoff: int = 10, ax=None):
        self.check_loaded("plot_vif")
        from qpoml.plotting import plot_vif

        data = self.context_tensor
        columns = self.context_features

        data = pd.DataFrame(data, columns=columns)

        plot_vif(data=data, cutoff=cutoff, ax=ax)

    ### POST EVALUATION ###

    def plot_results_regression(self, feature_name:str, which:list, ax = None, upper_lim_factor:float=1.025, font_scale:float=1.15):
        self.check_evaluated("plot_results_regression")
        from qpoml.plotting import plot_results_regression

        regression_x, regression_y, _, = self.results_regression(feature_name=feature_name, which=which)
        
        unit = self.units[feature_name]

        plot_results_regression(regression_x=regression_x, regression_y=regression_y, y_test=None, predictions=None, feature_name=feature_name, unit=unit, which=None, ax=ax, upper_lim_factor=upper_lim_factor, font_scale=font_scale)

    def plot_feature_importances(self, model, kind:str='tree-shap', style:str='bar', ax=None, cut:float=2, sigma:float=2.576, hline:bool=False, save_path:str=None):
        
        self.check_evaluated("plot_feature_importances")
        from qpoml.plotting import plot_feature_importances

        model = self.evaluated_model
        X_test = self.X_test
        y_test = self.y_test

        feature_names = self.context_features

        mean_importances_df, importances_df = self.feature_importances(feature_names=feature_names, kind=kind)

        plot_feature_importances(model=model, X_test=X_test, y_test=y_test, feature_names=feature_names, 
                                    kind=kind, style=style, ax=ax, cut=cut, sigma=sigma, 
                                    mean_importances_df=mean_importances_df, importances_df=importances_df, hline=hline)

        if save_path is not None: 
            importances_df.to_csv(save_path, index=False)

    # FIX FOLD PERFORMANCE ! #

    # FIX THISS!@!@!!!!
    def plot_fold_performance(self, statistic: str = "mae", ax=None):
        r"""

        FIX THIS!!! WILL NOT WORK GIVEN NEW WORKFLOW!!

        _Class method for visualizing predictive performance across different folds of test data_
        Parameters
        ----------
        statistic : `str`
            Either 'mse' for mean squared error, or 'mae' for median absolute error
        Returns
        -------
        """

        self.check_evaluated("plot_fold_performance")
        import matplotlib.pyplot as plt
        import seaborn as sns
        

        if wh1: 
            plt.style.use('/ar1/PROJ/fjuhsd/personal/thaddaeus/github/QPOML/qpoml/stylish.mplstyle')
        else: 
            plt.style.use('/mnt/c/Users/Research/Documents/GitHub/QPOML/qpoml/stylish.mplstyle')
            
        sns.set_context('paper')

        internal = False
        if ax is None:
            fig, ax = plt.subplots()
            internal = True

        measure = self.performance_statistics()
        measure = measure[statistic]

        folds = list(range(len(measure)))

        temp_df = pd.DataFrame(
            np.array([folds, measure]).T, columns=["fold", "measure"]
        )

        plt.plot(folds, measure, "-o", ms=4)
        ax.set_xlabel("Test Fold")
        ax.set_ylabel("Model " + statistic)
        ax.tick_params(bottom=True, labelbottom=False)
        plt.tick_params(axis="x", which="minor", bottom=False, top=False)

        if internal:
            plt.tight_layout()
            plt.show()

    #### CLASSIFICATION ####

    # FIX THIS!@!!!!!!
    def roc_and_auc(self, fold:int=None): 
        r'''
        
        FIX THIS SINCE WE'RE NOT DOING K-FOLD LIKE I WAS ORIGINALLY!!!! 

        Notes
        -----
        - Portions of this routine were taken from an sklearn documentation example that can be found at this [link](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html?highlight=roc+curve)
        '''

        from qpoml.utilities import roc_and_auc
        from sklearn.metrics import roc_curve, auc
        self.check_evaluated('roc_and_auc')

        if self.classification_or_regression=='classification': 
            predictions = self.predictions
            y_test = self.y_test 

            std_tpr = None
            std_auc = None

            if len(predictions)>2 and fold is None: 
                mean_fpr = np.linspace(0, 1, 100)
                tprs = []
                aucs = []
                
                for i in range(len(predictions)): 
                    true = y_test[i]
                    pred = predictions[i]

                    fpr, tpr, _ = roc_curve(true, pred)
                    if len(tpr)==np.sum(np.isfinite(tpr)):
                        auc_value = auc(fpr, tpr)

                        interp_tpr = np.interp(mean_fpr, fpr, tpr)
                        #print(interp_tpr)
                        interp_tpr[0] = 0.0
                        tprs.append(interp_tpr)

                        aucs.append(auc_value)

                mean_tpr = np.mean(tprs, axis=0)
                mean_tpr[-1] = 1.0
                std_tpr = np.std(tprs, axis=0)
                mean_auc = auc(mean_fpr, mean_tpr)
                std_auc = np.std(aucs)

                fpr = mean_fpr
                tpr = mean_tpr
                auc = mean_auc

            else: 
                fpr, tpr, auc = roc_and_auc(y_test[fold], predictions[fold])

            return fpr, tpr, std_tpr, auc, std_auc

        else: 
            raise Exception('')

    def plot_confusion_matrix(self, ax=None, cbar:bool=False, labels:list=None):
        self.check_evaluated('plot_confusion_matrix')
        from qpoml.plotting import plot_confusion_matrix

        if self.classification_or_regression == 'classification':
            
            y_test = self.y_test
            predictions = self.predictions 
            
            ax = plot_confusion_matrix(y_test, predictions, ax=ax, cbar=cbar, labels=labels)
            return ax 

        else: 
            raise Exception('')

    ## GOTCHAS ##
    
    def check_loaded(self, function: str):
        if not self.loaded:
            raise Exception(
                "collection must be loaded before "
                + function
                + "() function can be accessed"
            )

    def check_evaluated(self, function: str):
        if not self.evaluated:
            raise Exception(
                "collection must be evaluated before "
                + function
                + "() function can be accessed"
            )

    def check_loaded_evaluated(self, function: str):
        self.check_loaded(function)
        self.check_evaluated(function)

    def dont_do_twice(self, function: str):
        if function == "load" and self.loaded:
            raise Exception(
                "the function "
                + function
                + "() has already been executed and this step cannot be repeated on the object now"
            )
        elif function == "evaluate" and self.evaluated:
            raise Exception(
                "the function "
                + function
                + "() has already been executed and this step cannot be repeated on the object now"
            )