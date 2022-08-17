import numpy
import pandas
import warnings
import numpy as np
import pandas as pd

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

        ### EVALUATE INITIALIZED  ###
        self.train_indices = None  # done
        self.test_indices = None  # done

        self.evaluation_approach = None  # done

        self.qpo_preprocess = None 
        self.context_preprocess = None # not really needed so not implemented yet 

        self.X_train = None  # done
        self.X_test = None  # done
        self.y_train = None  # done
        self.y_test = None  # done

        self.train_observationIDs = None 
        self.test_observationIDs = None

        self.evaluated_models = None  # done
        self.predictions = None  # done

    ## LOAD ##

    def load(
        self,
        qpo_csv: str,
        context_csv: str,
        qpo_preprocess: dict,
        context_preprocess,
        context_type: str='scalar',
        spectrum_approach: str = "by-row",
        units:dict=None, 
        rebin: int = None) -> None:

        r"""
        _Class method for loading collection object_

        Parameters
        ----------

        qpo_csv : `str`
            File path to correctly formatted csv file with QPO information

        context_csv : `str`
            File path to correctly formatted csv file with context information

        context_type : `str`
            If it is set to "spectrum" preprocessing will handle context input as spectrum

        context_preprocess : `dict` or `str`
            Fix this

        qpo_preprocess : `dict` or `str`
            Fix this

        rebin : `int`
            Defaults to `None`. If set to integer, spectrum will be rebinned such that it contains `rebin` channels.


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

        if context_type == "spectrum":
            self.context_is_spectrum = True

            context_tensor = context_tensor.astype(float)

            spectral_ranges = []
            spectral_centers = []

            for col in context_features:
                col_list = col.split("_")
                spectral_centers.append(float(col_list[0]))
                col_list = col_list[1].split("-")
                spectral_ranges.append((col_list[0], col_list[1]))

            spectral_centers = np.array(spectral_centers).astype(float)

            if rebin is not None:

                from scipy.stats import binned_statistic

                lows = np.array([i[0] for i in spectral_ranges]).astype(float)
                highs = np.array([i[1] for i in spectral_ranges]).astype(float)

                rebinned_lows, _, _ = binned_statistic(
                    spectral_centers, lows, "min", bins=rebin
                )
                rebinned_highs, _, _ = binned_statistic(
                    spectral_centers, highs, "max", bins=rebin
                )

                spectral_ranges = [
                    [i, j] for i, j in zip(rebinned_lows, rebinned_highs)
                ]
                context_tensor = np.array(
                    [
                        binned_statistic(spectral_centers, i, "sum", bins=rebin)[0]
                        for i in context_tensor
                    ]
                )
                spectral_centers, _, _ = binned_statistic(
                    spectral_centers, spectral_centers, "mean", bins=rebin
                )

                temp_one = [str(i) + "_" for i in spectral_centers]

                # lol this is an insane list comprehension
                context_features = [
                    temp_one[index]
                    + str(spectral_ranges[index][0])
                    + "-"
                    + str(spectral_ranges[index][1])
                    for index in range(len(spectral_centers))
                ]

            ### NORMALIZE CONTEXT SPECTRUM ###
            temp_tensor = None
            if spectrum_approach == "by-row" or spectrum_approach == "as-is":
                temp_tensor = context_tensor
            elif spectrum_approach == "by-column":
                temp_tensor = np.transpose(context_tensor)
            else:
                raise Exception("")
            for index, arr in enumerate(temp_tensor):
                arr, (_, _, _) = preprocess1d(arr, context_preprocess)
                temp_tensor[index] = arr
            if spectrum_approach == "by-column":
                context_tensor = np.transpose(temp_tensor)
            else:
                context_tensor = temp_tensor

            self.spectral_ranges = spectral_ranges
            self.spectral_centers = spectral_centers

        else:
            transposed = np.transpose(context_tensor)
            for index, arr in enumerate(transposed):
                arr, (_, _, _) = preprocess1d(arr, context_preprocess[context_features[index]])
                transposed[index] = arr

            context_tensor = np.transpose(transposed)

        ### QPO ###

        num_qpos = []

        qpo_df = pd.read_csv(qpo_csv)
        qpo_features = [i for i in list(qpo_df) if i not in self.qpo_reserved_words]

        max_simultaneous_qpos = Counter(qpo_df["observation_ID"]).most_common(1)[0][1]
        max_length = max_simultaneous_qpos * len(qpo_features)
        qpo_tensor = []

        qpo_preprocess1d_tuples = {}

        for qpo_feature in qpo_features:
            if qpo_feature not in self.qpo_reserved_words:  # reserved QPO words
                
                modified, preprocess1d_tuple = preprocess1d(x=qpo_df[qpo_feature], preprocess=qpo_preprocess[qpo_feature])

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

        ### UPDATE ATTRIBUTES ###
        self.observation_IDs = observation_IDs
        self.num_qpos = num_qpos
        self.context_tensor = context_tensor
        self.qpo_tensor = qpo_tensor
        self.max_simultaneous_qpos = max_simultaneous_qpos
        self.context_features = context_features
        self.qpo_features = qpo_features

        self.units = units

        self.loaded = True

    ## EVALUATE ##

    def evaluate(
        self,
        model,
        evaluation_approach: str,
        test_proportion: float = 0.1,
        folds: int = None,
        repetitions: int = None, 
        hyperparameter_dictionary:dict=None) -> None:
        r"""
        _Evaluate an already initiated and loaded model_

        Parameters
        ----------

        model : `object`
            An initialized regressor object from another class, e.g. sklearn. It will be cloned and then have parameters reset, so it's okay (it's actually nesessary) that it is initalized

        evaluation_approach : `str`
            Can be `default` or `k-fold` ... Fix this!

        test_proportion : `float`
            Default is `0.1`; proportion of values to reserve for test set

        folds : `int`
            Default is `None`; if set to some integer, the model will be validated via K-Fold validation, with `K=folds`

        Returns
        -------
        """

        self.check_loaded("evaluate")
        self.dont_do_twice("evaluate")

        import sklearn 
        from sklearn.model_selection import KFold, train_test_split

        random_state = self.random_state
        context_tensor = self.context_tensor
        qpo_tensor = self.qpo_tensor
        observation_IDs = np.array(self.observation_IDs)

        train_indices = []
        test_indices = []

        evaluated_models = []
        predictions = []

        X_train = []
        X_test = []
        y_train = []
        y_test = []

        train_observation_IDs = []
        test_observation_IDs = []

        if evaluation_approach == "k-fold" and folds is not None:

            if repetitions is None:
                kf = KFold(n_splits=folds)

            else:
                from sklearn.model_selection import RepeatedKFold

                kf = RepeatedKFold(
                    n_splits=folds, n_repeats=repetitions, random_state=random_state
                )

            split = list(kf.split(context_tensor))

            for tr, te in split: 
                train_indices.append(tr)
                test_indices.append(te)

            #train_indices = np.array(train_indices).astype(int)
            #test_indices = np.array(test_indices).astype(int)

            for train_indices_fold, test_indices_fold in zip(train_indices, test_indices):
                X_train_fold = np.array(context_tensor[train_indices_fold])
                X_test_fold = np.array(context_tensor[test_indices_fold])
                y_train_fold = np.array(qpo_tensor[train_indices_fold])
                y_test_fold = np.array(qpo_tensor[test_indices_fold])
                
                local_model = sklearn.base.clone(model)

                if hyperparameter_dictionary is not None:
                    local_model = local_model.set_params(**hyperparameter_dictionary)
                
                local_model.fit(X_train_fold, y_train_fold)
                predictions.append(local_model.predict(X_test_fold))
                evaluated_models.append(local_model)

                X_train.append(X_train_fold)
                X_test.append(X_test_fold)
                y_train.append(y_train_fold)
                y_test.append(y_test_fold)

                train_observation_IDs.append(observation_IDs[train_indices_fold])
                test_observation_IDs.append(observation_IDs[test_indices_fold])

        elif evaluation_approach == "default":

            train_indices_fold, test_indices_fold = train_test_split(np.arange(0, len(qpo_tensor), 1).astype(int), test_size=test_proportion, random_state=random_state)

            X_train_fold = context_tensor[train_indices_fold]
            X_test_fold = context_tensor[test_indices_fold]
            y_train_fold = qpo_tensor[train_indices_fold]
            y_test_fold = qpo_tensor[test_indices_fold]

            local_model = sklearn.base.clone(model)

            if hyperparameter_dictionary is not None:
                local_model = local_model.set_params(**hyperparameter_dictionary)

            local_model.fit(X_train_fold, y_train_fold)
            predictions.append(local_model.predict(X_test_fold))
            evaluated_models.append(local_model)

            X_train.append(X_train_fold)
            X_test.append(X_test_fold)
            y_train.append(y_train_fold)
            y_test.append(y_test_fold)

            train_indices.append(train_indices_fold)
            test_indices.append(test_indices)

            train_observation_IDs.append(observation_IDs[train_indices_fold])
            test_observation_IDs.append(observation_IDs[test_indices_fold])

        else:
            raise Exception("")

        self.evaluation_approach = evaluation_approach

        self.train_indices = train_indices
        self.test_indices = test_indices

        self.evaluated_models = evaluated_models
        self.predictions = predictions

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.train_observation_IDs = train_observation_IDs
        self.test_observation_IDs = test_observation_IDs

        # future idea: let users stratify by more than just internal qpo count

        ### UPDATE ATTRIBUTES ###

        self.evaluated = True

    # SUPERFLUOUS #

    # what does sandman mean in the old slang? e.g. in hushaby song

    ## UTILITIES ##

    def get_performance_statistics(self):
        r"""
        _Return model performance statistics_

        Parameters
        ----------

        Returns
        -------

        statistics : `dict`
            Dictionary of performance statistics. Currently contains `mae` and `mse`
        """

        self.check_loaded_evaluated("performance_statistics")

        from sklearn.metrics import mean_absolute_error, mean_squared_error

        statistics = {}

        predictions = self.predictions
        y_test = self.y_test

        mae = None
        mse = None

        if len(predictions) == 1:
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

            statistics["mse"] = mse
            statistics["mae"] = mae

        return statistics

    def gridsearch(self, model, parameters: dict, n_jobs: int = None):
        r"""
        _Run five fold exhaustive grid search for hyperparameter tuning_

        Parameters
        ----------

        model :

        parameters :

        n_jobs :

        Returns
        -------

        """

        self.check_loaded("grid_search")

        from sklearn.model_selection import GridSearchCV

        if n_jobs is None:
            clf = GridSearchCV(model, parameters, scoring='neg_mean_absolute_error')
        else:
            clf = GridSearchCV(model, parameters, n_jobs=n_jobs, scoring='neg_mean_absolute_error')

        clf.fit(self.context_tensor, self.qpo_tensor)

        results = clf.cv_results_

        scores = np.array(results["mean_test_score"])
 
        stds = results['std_test_score']
        params = results["params"]

        sort_idx = np.argsort(results['rank_test_score'])
        best_params = clf.best_params_

        scores = np.array(scores)[sort_idx]
        stds = np.array(stds)[sort_idx]
        params = np.array(params)[sort_idx]
        
        return scores, stds, params, best_params

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

    def results_regression(self, feature_name:str, which: list, fold: int = None):

        r'''
        
        Arguments
        ---------

        feature_name : str
            Since preprocessing is handled under the hood within the collection class, users won't have access to their preprocess1d tuples. Thus, they need to also give the name of the feature (e.g. 'frequency') so this method can locate a saved copy of the preprocess1d tuple that matches with an output QPO feature that was registered when the collection object was loaded.

        '''


        from qpoml.utilities import results_regression

        self.check_evaluated("feature_importances")

        model = self.evaluated_models
        predictions = self.predictions
        y_test = self.y_test

        if self.evaluation_approach == "k-fold" and fold is not None:
            model = model[fold]
            predictions = predictions[fold]
            y_test = y_test[fold]

        else:
            model = model[0]
            predictions = predictions[0]
            y_test = y_test[0]

        regression_x, regression_y, linregress_result = results_regression(y_test=y_test, predictions=predictions, which=which, 
                                                                   preprocess1d_tuple=self.qpo_preprocess1d_tuples[feature_name])

        return regression_x, regression_y, linregress_result

    def feature_importances(self, feature_names: list, kind: str = "kernel-shap", fold: int = None):
        r"""
        fold is if the user previously chose to do k-fold cross validation, 0 index of models to select feature importances from
        """

        from qpoml.utilities import feature_importances

        self.check_evaluated("feature_importances")

        model = self.evaluated_models
        X_test = self.X_test
        y_test = self.y_test

        if self.evaluation_approach == "k-fold" and fold is not None:
            model = model[fold]
            X_test = X_test[fold]
            y_test = y_test[fold]

        else:
            model = model[0]
            X_test = X_test[0]
            y_test = y_test[0]

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

    def plot_results_regression(self, feature_name:str, which:list, ax = None, upper_lim_factor:float=1.025, fold:int = None):
        
        self.check_evaluated("plot_results_regression")
        from qpoml.plotting import plot_results_regression

        y_test = self.y_test
        predictions = self.predictions

        if self.evaluation_approach == "k-fold" and fold is not None:
            predictions = predictions[fold]
            y_test = y_test[fold]

        else:
            predictions = predictions[0]
            y_test = y_test[0]

        regression_x, regression_y, _, = self.results_regression(feature_name=feature_name, which=which, fold=fold)
        
        unit = self.units[feature_name]

        plot_results_regression(regression_x=regression_x, regression_y=regression_y, y_test=None, predictions=None, feature_name=feature_name, unit=unit, which=None, ax=ax, upper_lim_factor=upper_lim_factor)

    def plot_feature_importances(self, model, fold:int=None, kind:str='kernel-shap', style:str='bar', ax=None, cut:float=2, sigma:float=2):
        
        self.check_evaluated("plot_feature_importances")
        from qpoml.plotting import plot_feature_importances

        model = self.evaluated_models
        X_test = self.X_test
        y_test = self.y_test
        predictions = self.predictions

        if self.evaluation_approach == "k-fold" and fold is not None:
            feature_names = self.context_features

            mean_importances_df, importances_df = self.feature_importances(feature_names=feature_names, kind=kind, fold=fold)

            plot_feature_importances(model=model, X_test=X_test, y_test=y_test, feature_names=feature_names, 
                                     kind=kind, style=style, ax=ax, cut=cut, sigma=sigma, 
                                     mean_importances_df=mean_importances_df, importances_df=importances_df)

        else: 
            print('error!')

    def plot_fold_performance(self, statistic: str = "mae", ax=None):
        r"""
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

        plt.style.use(
            "https://gist.githubusercontent.com/thissop/44b6f15f8f65533e3908c2d2cdf1c362/raw/fab353d758a3f7b8ed11891e27ae4492a3c1b559/science.mplstyle"
        )

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
