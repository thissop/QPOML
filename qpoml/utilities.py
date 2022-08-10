from tkinter import N
import numpy as np
import numpy 
import pandas as pd
import pandas 
import warnings
 
## METHODS ## 

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

def compare_models(first_scores:numpy.array, second_scores:numpy.array, n_train:int, n_test:int, approach:str, rope:list=[[-0.01, 0.01], 0.95]):
    r'''
    Parameters
    ----------
    first_scores : 

    second_scores : 

    num_train : 

    num_test : 

    approach :

    rope : list
         
    
    '''
    from scipy.stats import t
    from qpoml.utilities import corrected_std, compute_corrected_ttest

    differences = np.array(first_scores) - np.array(second_scores)

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

def calculate_vif(data:pandas.DataFrame): 
    from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
    
    temp = data.select_dtypes(['number'])

    vif_df = pd.DataFrame()
    vif_df['VIF'] = [vif(temp.values, i) for i in range(temp.shape[1])]
    vif_df['Column'] = list(temp)

    vif_df.sort_values('VIF', ascending=True)

    return vif_df 

### POST EVALUATION ### 

def results_regression(y_test:numpy.array, predictions:numpy.array, which:list, preprocess1d_tuple:tuple, 
                       regression_x=None, regression_y=None): # will work best with result vectors of my design 
    r'''
    
    Execute "results regression" on predictions based on their true values.  

    Parameters
    ----------

    y_test : numpy.array 
        True values to compare predicted ones to 

    predictions : numpy.array
        Array of predicted values 

    which : list 
        List of pythonic indices corresponding to the value(s) in the `y_test`/`predictions` vectors upon which `results_regression` should be run; e.g. if QPO vector is `[frequency,width,amplitude]`, `what=[0]` will run `results_regression` on frequency only because only the zeroth item in every predicted vector, the QPO frequency in this case, will be concatenated to the flattened arrays for regression. Similarly, if the QPO prediction vectors followed the form `[First QPO Frequency, First QPO Width, Second QPO Frequency, Second QPO Width]`, `what=[0,2]` would compute `results_regression` on only a concatenated array of First and Second QPO Frequencies.      

    preprocess1d_tuple : tuple 
        the second thing returned by preprocess1d; only one is provided for both x and y because both should be un-processed in the same way. 

    Returns
    -------

    variable_name : type
        Description 

    regression_x : numpy.array 
        Flattened (from potentially concatenated array) of true values 

    regression_y : numpy.array 
        Flattened (from potentially concatenated array) of predicted values 
    
    line_tuple : tuple 
        Returns `(m,b)`, i.e. best fit slope and intercept

    stats_tuple : tuple   
        Returns `(r, pval, stderr, intercept_stderr)`; see documentation for `scipy.stats.linregress`

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
    
def feature_importances(model, X_test, y_test, feature_names, kind:str='kernel-shap'):
    
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
            raise Exception('')
                
    elif kind=='permutation': 
        from sklearn.inspection import permutation_importance

        permutation_importances = permutation_importance(estimator=model, X=X_test, y=y_test)
        
        mean_importances = permutation_importances.importances_mean 

        sort_idx = np.argsort(mean_importances)[::-1]
        importances_df = pd.DataFrame(permutation_importances.importances[sort_idx].T, columns=feature_names[sort_idx])

    elif kind=='kernel-shap' or kind=='tree-shap': 
        
        if kind=='kernel-shap':
            explainer = shap.Explainer(model,X_test, check_additivity=False)
        else: 
            explainer = shap.TreeExplainer(model, data=X_test, check_additivity=False)

        shap_values = explainer(X_test).values.T

        absolute_arrays = np.array([np.abs(i) for i in shap_values])

        if len(absolute_arrays.shape)>2: 
            absolute_arrays = np.concatenate(absolute_arrays, axis=1)

        mean_importances = np.mean(absolute_arrays, axis=1)
        sort_idx = np.argsort(mean_importances)[::-1]

        importances_df = pd.DataFrame(np.transpose(absolute_arrays[sort_idx]), columns=feature_names[sort_idx])

    else: 
        raise Exception('')

    sort_idx = sort_idx.astype(int)
    mean_importances = mean_importances[sort_idx]
    feature_names = feature_names[sort_idx]

    mean_importances_df = pd.DataFrame()
    for index in range(len(feature_names)): 
        mean_importances_df[feature_names[index]] = [mean_importances[index]]

    return mean_importances_df, importances_df 
    
def confusion_matrix(y_test:numpy.array, predictions:numpy.array): 
    from sklearn.metrics import confusion_matrix, accuracy_score

    y_test = np.array(y_test).flatten()
    predictions = np.array(predictions).flatten()  

    cm = confusion_matrix(y_test, predictions)
    acc = accuracy_score(y_test, predictions)

    return cm, acc

### MULTI-MODEL RELATED ###

def bulk_load(n:int, qpo_csv:str, context_csv:str, qpo_preprocess:str, context_preprocess, context_type:str='scalar', spectrum_approach:str='by-row', rebin:int=None):
    r'''
    initialize multiple identical qpoml collection objects
    '''

    from qpoml import collection 

    loaded_collections = []

    for i in range(n): 
        c = None 
        c = collection()
        c.load(qpo_csv=qpo_csv, context_csv=context_csv, context_type=context_type, context_preprocess=context_preprocess,
               qpo_preprocess=qpo_preprocess, spectrum_approach=spectrum_approach, rebin=rebin)

        loaded_collections.append(c)
    
    return loaded_collections

### XSPEC RELATED ###
'''
def calculate_hardness(spectrum:xspec.Spectrum, soft_range:list, hard_range:list, calculation:str='proportion'):
    spectrum.ignore('**-'+str(soft_range[0]))
    spectrum.ignore(str(soft_range[1])+'-**')
    soft_sum = np.sum(spectrum.values)

    spectrum.notice('**-**')
    spectrum.ignore('**-'+str(hard_range[0]))
    spectrum.ignore(str(hard_range[1])+'-**')
    hard_sum = np.sum(spectrum.values)

    if calculation == 'proportion':
        return hard_sum/(soft_sum+hard_sum) 
    elif calculation == 'ratio':
        return hard_sum/soft_sum
'''

### TO DO ###
r'''

# these are not that important to me right now, but should be eventually (because convenience can make the package marketable) ... I think they can't be static...they need to be instance 

def remove_by_vif(cutoff_value): # remove columns from context with vif more than cutoff ... note that multicollinearity is not a concern for pure accuracy, mainly a concern when dealing with feature importances, which is important in elucidating useful takeaways; only applied to context 
    self.check_loaded('remove_by_vif')

    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

    vif_info = pd.DataFrame()
    vif_info['VIF'] = [vif(X.values, i) for i in range(X.shape[1])]
    vif_info['Column'] = X.columns
    vif_info.sort_values('VIF', ascending=True)

    mask = np.where(vif_info['VIF']<5)[0]

    ols_cols = vif_info['Column'][mask]

def remove_from_dendrogram(cutoff_value): # rename? 
    pass 

def pca_transform(): # once these happen, context_df and arrays are changed to place holder names with transformed vectors;  only applied to context
    self.check_loaded('pca_transform') # https://github.com/thissop/MAXI-J1535/blob/main/code/machine-learning/December-%202021-2022/very_initial_sanity_check.ipynb

def mds_transform(): # only applied to context
    self.check_loaded('mds_transform')



'''