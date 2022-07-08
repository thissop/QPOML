import numpy as np
import numpy 
import pandas as pd
import pandas 
import warnings
 
## METHODS ## 

### BASIC ###

def preprocess1d(x, preprocess): 
    
    r'''
    preprocess : list, str
        If it's a list, then preprocess[0] is min value for norm, and ...[1] is max value for norm 
    '''
    
    if type(preprocess) is list: 
        min_value = preprocess[0]
        max_value = preprocess[1]
        modified = (x-min_value)/(max_value-min_value) 
        modified = modified*(1 - 0.1) + 0.1 # so it will be output as 0.1-1 range 
    elif type(preprocess) is str: 
        if preprocess == 'as-is': 
            modified = x 
        elif preprocess == 'normalize': 
            min_value = np.min(x)
            max_value = np.max(x)
            modified = (x-min_value)/(max_value-min_value)
            modified = modified*(1 - 0.1) + 0.1
        elif preprocess == 'standardize': 
            mean = np.mean(x)
            sigma = np.std(x)
            modified = (x-mean)/sigma
        elif preprocess == 'median': 
            modified = x/np.median(x)
        else: 
            raise Exception('')
    
    
    return modified

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
        t_stat, p_val = compute_corrected_ttest(differences=differences, df=dof, n_train=n_train, n_test=n_test)
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

def compute_corrected_ttest(differences, df, n_train, n_test):
    r'''
    _Computes right-tailed paired t-test with corrected variance._

    Parameters
    ----------
    differences : array-like of shape (n_samples,)
        Vector containing the differences in the score metrics of two models.
    df : int
        Degrees of freedom.
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

def results_regression(y_test:numpy.array, predictions:numpy.array, which:list, 
                       regression_x:numpy.array=None, regression_y:numpy.array=None): # will work best with result vectors of my design 
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
    
    if regression_x is None and regression_y is None: 
        regression_x = np.transpose(np.array(y_test))
        regression_y = np.transpose(np.array(predictions))

        regression_x, regression_y = (i[which] for i in (regression_x, regression_y)) 

    regression_x = regression_x.flatten().astype(float)
    regression_y = regression_y.flatten().astype(float)

    linregress_result = linregress(regression_x, regression_y) 

    return regression_x, regression_y, linregress_result 
    
def feature_importances(model, X_test, y_test, feature_names:list, kind:str='kernel-shap'):
    
    import shap
    
    importances_df = None
    shap_values = None
    feature_importances_arr = None
    sort_idx = None 

    if kind=='default':
        if hasattr(model, 'feature_importances_'): 
            feature_importances_arr = model.feature_importances_
                
    elif kind=='permutation': 
        
        from sklearn.inspection import permutation_importance

        permutation_importances = permutation_importance(estimator=model, X=X_test, y=y_test)
        feature_importances_arr = permutation_importances['importances_mean']

        sort_idx = np.argsort(feature_importances_arr)[::-1]

        importances_df = pd.DataFrame(permutation_importances.importances[sort_idx].T, columns=feature_names[sort_idx])
        
    elif kind=='kernel-shap': 
        warnings.warn('need to implement check to see if model is supported')
        explainer = shap.KernelExplainer(model.predict, X_test)
        shap_values = explainer.shap_values(X_test)
        shap_values = np.array(shap_values).T
        feature_importances_arr = np.array([np.mean(np.abs(i)) for i in shap_values])

    elif kind=='tree-shap': 
        warnings.warn('need to check if possible!')
        explainer = shap.TreeExplainer(model)
        shap_values = np.array(explainer.shap_values(X_test))
        shap_values = np.array(shap_values).T
        feature_importances_arr = np.array([np.mean(np.abs(i)) for i in shap_values])

    else: 
        raise Exception('')

    if sort_idx is None: 
        sort_idx = np.argsort(feature_importances_arr)[::-1]

    sort_idx = sort_idx.astype(int)

    feature_importances_arr = feature_importances_arr[sort_idx]
    feature_names = np.array(feature_names)[sort_idx]

    if kind=='tree-shap' or kind=='kernel-shap': 
        importances_df = pd.DataFrame()
        for index in range(len(feature_names)): 
            importances_df[feature_names[index]] = [feature_importances_arr[index]]

    return feature_importances_arr, feature_names, importances_df
    
def confusion_matrix(y_test:numpy.array, predictions:numpy.array): 
    from sklearn.metrics import confusion_matrix, accuracy_score

    y_test = np.array(y_test).flatten()
    predictions = np.array(predictions).flatten()  

    cm = confusion_matrix(y_test, predictions)
    acc = accuracy_score(y_test, predictions)

    return cm, acc

## CLASSES ## 

class eurostep_model: 

    def __init__(self, regressor, regressor_name:str, classifier, classifier_name:str, context_tensor:np.array, qpo_tensor:np.array) -> None:
        self.regressor = regressor 
        self.regressor_name = regressor_name
        self.classifier = classifier
        self.classifier_name = classifier_name 

    def return_classifier(self):

        # check if model has been evaluated first!
        return self.regressor

    def return_classifier(self): 
        # check if model has been evaluated first! 
        return self.classifier

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