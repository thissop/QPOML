import numpy as np
import numpy 
import pandas as pd
import pandas 
import warnings
 
## METHODS ## 
 
def gridsearch(model, observation_IDs, X, y, gridsearch_dictionary, class_or_reg, stratify, folds, num_qpo_features:int, random_state, repetitions:int=None, multiclass:bool=False): 
    r'''
    
    Arguments
    ---------

    stratify : 
        - if regression, stratify can be two things: 'num_qpo', or a two colummn dataframe in itself with the columns 'observation_ID' and 'class', with 'class' being something categorical that can be split on. 
        - if classification, it's just True or False

    Notes
    -----
        - need to let it save results! 
        - incorporate roc and auc to best model that runs after grid search so I can have the the five splits curves and their average still in the paper!!
    '''

    from itertools import product
    import os 
    import sklearn 
    from sklearn.model_selection import KFold, StratifiedKFold, RepeatedStratifiedKFold, RepeatedKFold, GridSearchCV
    from qpoml.utilities import roc_and_auc

    n_jobs = np.min([np.product([len(i) for i in gridsearch_dictionary.values()]), os.cpu_count()-1])

    FPRs = []
    TPRs = []
    auc_scores = []

    if stratify is not None and type(stratify) is bool: 
        if not stratify: 
            stratify = None

    best_model_fold_scores = []
    if class_or_reg == 'classification':

        cv = None
        
        if stratify: 
            if repetitions is not None: 
                cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repetitions, random_state=random_state)
            else: 
                cv = StratifiedKFold(n_splits=folds, n_repeats=repetitions, random_state=random_state)

        else: 
            if repetitions is not None: 
                cv = RepeatedKFold(n_splits=folds, n_repeats=repetitions, random_state=random_state)
            else: 
                cv = KFold(n_splits=folds, random_state=random_state)

        if multiclass: 
            clf = GridSearchCV(model, gridsearch_dictionary, cv=cv, n_jobs=n_jobs)
        else:    
            clf = GridSearchCV(model, gridsearch_dictionary, scoring='f1', cv=cv, n_jobs=n_jobs)

        clf.fit(X, y)

        results = clf.cv_results_

        scores = np.array(results["mean_test_score"])

        stds = results['std_test_score']
        params = results["params"]

        sort_idx = np.argsort(results['rank_test_score'])
        best_params = clf.best_params_

        scores = np.array(scores)[sort_idx]
        stds = np.array(stds)[sort_idx]
        params = np.array(params)[sort_idx]

        best_model_config = sklearn.base.clone(model)
        best_model_config = best_model_config.set_params(**best_params)

        if not multiclass: 

            if stratify: 
                for train_index, test_index in cv.split(X, y):
                    temp_model = sklearn.base.clone(best_model_config)
                    temp_model.fit(X[train_index], y[train_index])
                    
                    predictions = temp_model.predict(X[test_index])
                    fpr, tpr, auc_score = roc_and_auc(y[test_index], predictions)
                    FPRs.append(fpr)
                    TPRs.append(tpr)
                    auc_scores.append(auc_score)

                    best_model_fold_scores.append(sklearn.metrics.f1_score(y[test_index], predictions))

            else: 
                for train_index, test_index in cv.split(X):
                    temp_model = sklearn.base.clone(best_model_config)
                    temp_model.fit(X[train_index], y[train_index])
                    
                    predictions = temp_model.predict(X[test_index])
                    fpr, tpr, auc_score = roc_and_auc(y[test_index], predictions)
                    FPRs.append(fpr)
                    TPRs.append(tpr)
                    auc_scores.append(auc_score)

                    best_model_fold_scores.append(sklearn.metrics.f1_score(y[test_index], predictions))

    else:

        if stratify is not None:
            qpos_per_obs = []
            for i in X: 
                num = int(len(np.where(i!=0.1)[0])/(num_qpo_features))
                qpos_per_obs.append(num)

            if repetitions is not None: 
                kf = RepeatedStratifiedKFold(n_splits=folds)

                if type(stratify) is bool:
                    if stratify is True:  
                        split = list(kf.split(X=X, y=qpos_per_obs)) 
                else: 
                    stratify_df = pd.DataFrame()
                    stratify_df['observation_ID'] = observation_IDs
                    stratify_df = stratify_df.merge(stratify, on='observation_ID') 
                    split = list(kf.split(X=X, y=stratify_df['class'])) 
            
            else: 
                kf = StratifiedKFold(n_splits=folds) 
                
                if type(stratify) is bool:
                    if stratify is True:  
                        split = list(kf.split(X=X, y=qpos_per_obs))
                
                else: 
                    stratify_df = pd.DataFrame()
                    stratify_df['observation_ID'] = observation_IDs
                    stratify_df = stratify_df.merge(stratify, on='observation_ID') 
                    split = list(kf.split(X=X, y=stratify_df['class'])) 

        else: 
            if repetitions is not None: 
                kf = RepeatedKFold(n_splits=folds, n_repeats=repetitions, random_state=random_state)
            else: 
                kf = KFold(n_splits=folds)
            
            split = list(kf.split(X))

        cv = [(tr, te) for tr, te in split]
        clf = GridSearchCV(model, gridsearch_dictionary, scoring='neg_mean_absolute_error', cv=cv, n_jobs=n_jobs)
        clf.fit(X, y)
        results = clf.cv_results_

        scores = np.array(results["mean_test_score"])

        stds = results['std_test_score']
        params = results["params"]

        sort_idx = np.argsort(results['rank_test_score'])
        best_params = clf.best_params_

        scores = np.abs(np.array(scores)[sort_idx])
        stds = np.array(stds)[sort_idx]
        params = np.array(params)[sort_idx]

        #processess = multiprocessing.cpu_count()-2
         # Best Regression Model on Folds
        for train_indices_fold, test_indices_fold in cv:
            X_train_fold = np.array(X[train_indices_fold])
            X_test_fold = np.array(X[test_indices_fold])
            y_train_fold = np.array(y[train_indices_fold])
            y_test_fold = np.array(y[test_indices_fold])
            
            local_model = sklearn.base.clone(model)

            local_model.set_params(**best_params)
            
            local_model.fit(X_train_fold, y_train_fold)

            score = sklearn.metrics.mean_absolute_error(y_test_fold, local_model.predict(X_test_fold)) # there, now not dependent on sklearn model.score function
            best_model_fold_scores.append(score)

    best_model_config = sklearn.base.clone(model)
    best_model_config = best_model_config.set_params(**best_params)

    return (best_model_config, best_params), (scores, stds, params, best_model_fold_scores), (FPRs, TPRs, auc_scores)
    
### BASIC ###

def preprocess1d(x, preprocess, range_low:float=0.1, range_high:float=1.0): 

    r'''
    
    range_low, range_high : float, float 
        feature range that inputs are mapped to. default is 0.1 to 1.0 

    '''

    if type(preprocess) is str: 
        
        if preprocess == 'as-is': 
            modified = x 
            return modified, ('as-is', None, None) 
        
        elif preprocess == 'normalize': 
            min_value = np.min(x)
            max_value = np.max(x)
            modified = (x-min_value)/(max_value-min_value)
            modified = modified*(range_high - range_low) + range_low
            return modified, ('normalize', min_value, max_value)
        
        elif preprocess == 'standardize': 
            mean = np.mean(x)
            sigma = np.std(x)
            modified = (x-mean)/sigma
            return modified, ('standardize', mean, sigma)

        elif preprocess == 'median': 
            median = np.median(x)
            modified = x/median
            return modified, ('median', median, None) 
        
        else: 
            raise Exception('')
    
    else: 
        try: 
            min_value = preprocess[0]
            max_value = preprocess[1]
            modified = (x-min_value)/(max_value-min_value) 
            modified = modified*(range_high - range_low) + range_low # so it will be output as 0.1-1 range 
            return modified, ('normalize', min_value, max_value)
        except Exception as e: 
            print(e)
    
def unprocess1d(modified, preprocess1d_tuple, range_low:float=0.1, range_high:float=1.0): 
    
    method_applied = preprocess1d_tuple[0]

    x = None 

    if method_applied == 'as-is': 
        x = modified 

    elif method_applied == 'normalize': 
        applied_max = preprocess1d_tuple[2]
        applied_min = preprocess1d_tuple[1]
        
        x = (((modified-range_low)/(range_high-range_low))*(applied_max-applied_min))+applied_min # sheeesh
        
    elif method_applied == 'standardize':
        mean = preprocess1d_tuple[1] 
        sigma = preprocess1d_tuple[2] 
        x = (modified*sigma)+mean 
    
    elif method_applied == 'median': 
        x = modified*preprocess1d_tuple[1] 

    else: 
        raise Exception('')

    return x 
 
### "BORROWED" ###

def pairwise_compare_models(model_names:list, model_score_arrays:list, n_train:int, n_test:int, save_dir:str, save_path:str=None, rope:list=[[-0.01, 0.01], 0.95], decimals=2):
    
    r'''
    
    only works on regression right now!
    
    '''

    if save_dir is not None and save_dir[-1]!='/':
        save_dir += '/'

    sorted_first_names = []
    sorted_second_names = [] 
    t_values = []
    p_values = []
    first_better_percents = []
    second_better_percents = []

    idx = np.argsort(np.mean(model_score_arrays, axis=1))
    model_names = np.array(model_names)[idx]
    model_score_arrays = np.array(model_score_arrays)[idx]

    model_n = len(model_names)
    # I am not sure what 
    for i, first_name in enumerate(model_names):   
        first_score_arr = model_score_arrays[i]   
        first_average = np.mean(first_score_arr)
        for j in range(i+1, model_n):   
            second_score_arr = model_score_arrays[j]  
            second_name = model_names[j]
            second_average = np.mean(second_score_arr)

            sorted_first_names.append(first_name)
            sorted_second_names.append(second_name)
            if first_average > second_average: 
                t_values.append(['-'])
                p_values.append(['-'])

            else: 
                t, p = compare_models(first_score_arr, second_score_arr, n_train=n_train, n_test=n_test, approach='frequentist', better='lower')
                t_values.append(round(t, 2))
                p_values.append(p)

            first_better, second_better, _, _ = compare_models(first_score_arr, second_score_arr, n_train=n_train, n_test=n_test, approach='bayesian', better='lower')

            first_better = round(100*first_better, 2)
            second_better = 100-first_better

            first_better_percents.append(first_better)
            second_better_percents.append(second_better)
    
    temp_columns = [sorted_first_names, sorted_second_names, t_values, p_values, first_better_percents, second_better_percents]
    temp_names = ['First Model Name', 'Second Model Name', 't', 'p', '% Chance First Better', '% Chance Second Better']
    
    pairwise_results_df = pd.DataFrame()
    for i in range(len(temp_columns)): 
        pairwise_results_df[temp_names[i]] = temp_columns[i]
    
    pairwise_results_df.round(decimals=decimals)

    if save_dir is None: 
        pairwise_results_df.to_csv(f'{save_path}.csv', index=False)
        pairwise_results_df.to_latex(f'{save_path}.tex', index=False)#, float_format="%.6f")
    else: 
        pairwise_results_df.to_csv(f'{save_dir}comparison_table.csv', index=False)
        pairwise_results_df.to_latex(f'{save_dir}comparison_table.tex', index=False)#, float_format="%.6f")

def compare_models(first_scores:numpy.array, second_scores:numpy.array, n_train:int, n_test:int, approach:str, better:str, rope:list=[[-0.01, 0.01], 0.95]):
    
    r'''
    Parameters
    ----------
    first_scores : 

    second_scores : 

    num_train : 

    num_test : 

    approach :

    rope : list

    better : 
        lower or higher
         
    NOTES
    -----

    FOR NOW THIS ONLY WORKS WITH comparison metric = mae (thus, regression). This sets better=higher for bayesian
    
    Do not even use this function alone ... only use with the pairwise_compare_models() wrapper 

    '''
    from scipy.stats import t
    from qpoml.utilities import corrected_std, compute_corrected_ttest

    order = np.argsort([np.mean(first_scores), np.mean(second_scores)])
    scores = np.array([first_scores, second_scores])
    if better == 'lower':
        scores = scores[order]
        differences = scores[1]-scores[0]

    elif better == 'higher':
        order = order[::-1]
        scores = scores[order]
        differences = scores[1]-scores[0]

    else: 
        raise Exception('')


    #differences = np.array(first_scores) - np.array(second_scores)

    dof = len(differences)-1

    if approach == 'frequentist': 
        t_stat, p_val = compute_corrected_ttest(differences=differences, n_train=n_train, n_test=n_test)
        return t_stat, p_val

    elif approach == 'bayesian': 
        posterior = t(dof, loc=np.mean(differences), scale=corrected_std(differences, n_train, n_test))
        first_better_than_second = 1 - posterior.cdf(0)
        second_better_than_first = 1-first_better_than_second

        if rope is None: 

            return first_better_than_second, second_better_than_first 

        else: 
            rope_interval = rope[0]
            rope_prob = posterior.cdf(rope_interval[1]) - posterior.cdf(rope_interval[0])
            cred_interval = list(posterior.interval(rope[1]))

            return first_better_than_second, second_better_than_first, rope_prob, cred_interval
    
    else: 
        raise Exception('')

def corrected_std(differences, n_train, n_test):
    r'''
    _Corrects standard deviation using Nadeau and Bengio's approach_

    Parameters
    ----------
    differences : ndarray of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    corrected_std : float
        Variance-corrected standard deviation of the set of differences.
    
    Notes
    -----
        Code borrowed from [here](https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html?highlight=statistical%20comparison%20models)
    '''
    
    # kr = k times r, r times repeated k-fold crossvalidation,
    # kr equals the number of times the model was evaluated
    kr = len(differences)
    corrected_var = np.var(differences, ddof=1) * (1 / kr + n_test / n_train)
    corrected_std = np.sqrt(corrected_var)
    return corrected_std

def compute_corrected_ttest(differences, n_train, n_test):
    r'''
    _Computes right-tailed paired t-test with corrected variance._

    Parameters
    ----------
    differences : array-like of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    n_train : int
        Number of samples in the training set.
    n_test : int
        Number of samples in the testing set.

    Returns
    -------
    t_stat : float
        Variance-corrected t-statistic.
    p_val : float
        Variance-corrected p-value.
    
    Notes
    -----
        Code borrowed from [here](https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html?highlight=statistical%20comparison%20models) 
    
    '''

    from scipy.stats import t

    df = n_test-1

    mean = np.mean(differences)
    std = corrected_std(differences, n_train, n_test)
    t_stat = mean / std
    p_val = t.sf(np.abs(t_stat), df)  # right-tailed t-test
    return t_stat, p_val

### POST LOAD ### 

def correlation_matrix(data:pandas.DataFrame): # <== I don't think this works for spectrums? 

    temp_df = data.select_dtypes(['number'])
    corr = temp_df.corr()
    cols = list(temp_df)
    
    return corr, cols

def dendrogram(data:pandas.DataFrame):
    from scipy.stats import spearmanr
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import squareform

    temp = data.select_dtypes(['number'])
    cols = list(temp)

    # below from sklearn documentation 

    # Ensure the correlation matrix is symmetric
    corr = spearmanr(temp).correlation
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)

    # We convert the correlation matrix to a distance matrix before performing
    # hierarchical clustering using Ward's linkage.
    distance_matrix = 1 - np.abs(corr)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))
    
    return corr, dist_linkage, cols 

# DONE
def calculate_vif(data:pandas.DataFrame): 
    r'''
    __Calculate VIFs for features in dataframe__
    '''
    from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
    
    temp = data.select_dtypes(['number'])

    vif_df = pd.DataFrame()
    vif_df['VIF'] = [vif(temp.values, i) for i in range(temp.shape[1])]
    vif_df['Column'] = list(temp)

    vif_df.sort_values('VIF', ascending=True)

    return vif_df 

### POST EVALUATION ### 

# DONE
def results_regression(y_test:numpy.array, predictions:numpy.array, which:list, preprocess1d_tuple:tuple, 
                       regression_x=None, regression_y=None):  
    r'''
        _Calculate "results regression" on relationship between true and predicted values_
        Arguments
        ---------

        y_test : np.array 
            Array of true values
        predictions : np.array
            Array of predicted values       
        which : list
            Index of which of those features should be included. E.g. if the maximum number of QPOs for an object under consideration is 3, then if you only wanted to calculate results regression on the first QPO for every predicted observation, you would set which=[0]. If you wanted to include all three QPOs, then you would set which=[0,1,2]
        preprocess1d_tuple : tuple
            Tuple returned by preprocess_1d for the feature        
        regression_x : np.array
            If you already have the regression values, skip everything else and just calculate the regression (same for regression_y, both need to be defined)

        regression_y : np.array 
            See regression_x

        Returns
        -------

        regression_x : np.array
            Combined array of all the true values accumulated from the indices defined by which for the feature 
        regression_y : np.array 
            Combined array of all the values predicted by the model and accumulated from the indices defined by which for the feature 
        linregress_result : 
            Result from scipy linear regression on regression_x and regression_y (r, pval, stderr, intercept_stderr)

        '''
    
    from scipy.stats import linregress 
    from qpoml.utilities import unprocess1d 
     
    if regression_x is None and regression_y is None: 
        regression_x = np.transpose(np.array(y_test))
        regression_y = np.transpose(np.array(predictions))

        regression_x, regression_y = (i[which] for i in (regression_x, regression_y)) 
        regression_x = unprocess1d(regression_x, preprocess1d_tuple)
        regression_y = unprocess1d(regression_y, preprocess1d_tuple)

    regression_x = regression_x.flatten().astype(float)
    regression_y = regression_y.flatten().astype(float)

    linregress_result = linregress(regression_x, regression_y) 

    return regression_x, regression_y, linregress_result 
    
def feature_importances(model, X_test, y_test, feature_names, kind:str='tree-shap'):
    
    import shap
    import pandas as pd
    
    importances_df = None 
    mean_importances_df = None 
    shap_values = None 
    sort_idx = None 

    feature_names = np.array(feature_names)

    if kind=='default':
        if hasattr(model, 'feature_importances_'): 
            mean_importances = model.feature_importances_
            sort_idx = np.argsort(mean_importances)[::-1]
        else: 
            raise Exception('Model does not have attribute feature_importances_. Try using "kernel-shap"')
                
    elif kind=='permutation': 
        from sklearn.inspection import permutation_importance

        permutation_importances = permutation_importance(estimator=model, X=X_test, y=y_test)
        
        mean_importances = permutation_importances.importances_mean 

        sort_idx = np.argsort(mean_importances)[::-1]
        importances_df = pd.DataFrame(permutation_importances.importances[sort_idx].T, columns=feature_names[sort_idx])

    elif kind=='kernel-shap' or kind=='tree-shap': 
        
        if kind=='kernel-shap':
            explainer = shap.Explainer(model, X_test, check_additivity=False)
        else: 
            try: 
                explainer = shap.TreeExplainer(model, data=X_test, check_additivity=False)
            except Exception as e: 
                explainer = shap.Explainer(model, X_test, check_additivity=False)

        shap_values = explainer(X_test).values.T

        absolute_arrays = np.array([np.abs(i) for i in shap_values])

        if len(absolute_arrays.shape)>2: 
            absolute_arrays = np.concatenate(absolute_arrays, axis=1)

        mean_importances = np.mean(absolute_arrays, axis=1)
        sort_idx = np.argsort(mean_importances)[::-1]

        importances_df = pd.DataFrame(np.transpose(absolute_arrays[sort_idx]), columns=feature_names[sort_idx])

    else: 
        raise Exception(f'{kind} is illegal for feature importance! Please choose different approach (options include "kernel-shap", "tree-shap", "permutation", and "default")')

    sort_idx = sort_idx.astype(int)
    mean_importances = mean_importances[sort_idx]
    feature_names = feature_names[sort_idx]

    mean_importances_df = pd.DataFrame()
    for index in range(len(feature_names)): 
        mean_importances_df[feature_names[index]] = [mean_importances[index]]

    return mean_importances_df, importances_df 
    
# DONE
def confusion_matrix(y_test:numpy.array, predictions:numpy.array): 
    r'''Wrapper for calculating confusion matrix and accuracy for classification'''
    from sklearn.metrics import confusion_matrix, accuracy_score

    y_test = np.array(y_test).flatten()
    predictions = np.array(predictions).flatten()  

    cm = confusion_matrix(y_test, predictions)
    acc = accuracy_score(y_test, predictions)

    return cm, acc

# DONE
def roc_and_auc(y_test:numpy.array, predictions:numpy.array): 
    r'''
    __Wrapper for calculating`ROC and AUC from y_test and predictions__
    '''
    
    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc 

    fpr, tpr, thresholds = roc_curve(y_test, predictions)
    auc_score = auc(fpr, tpr)

    return fpr, tpr, auc_score
