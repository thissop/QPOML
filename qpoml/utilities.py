import numpy as np
import numpy 
import pandas as pd
import pandas 
import warnings

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
    elif type(preprocess) is str: 
        if preprocess == 'as-is': 
            modified = x 
        elif preprocess == 'normalize': 
            min_value = np.min(x)
            max_value = np.max(x)
            modified = (x-min_value)/(max_value-min_value)
        elif preprocess == 'standardize': 
            mean = np.mean(x)
            sigma = np.std(x)
            modified = (x-mean)/sigma
        elif preprocess == 'median': 
            modified = x/np.median(x)
        else: 
            raise Exception('')
    
    
    return modified

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

def results_regression(y_test:numpy.array, predictions:numpy.array, what:list): # will work best with result vectors of my design 
    r'''
    
    Execute "results regression" on predictions based on their true values.  

    Parameters
    ----------

    y_test : numpy.array 
        True values to compare predicted ones to 

    predictions : numpy.array
        Array of predicted values 

    what : list 
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
    
    regression_x = np.transpose(np.array(y_test))
    regression_y = np.transpose(np.array(predictions))

    regression_x, regression_y = (i[which] for i in (regression_x, regression_y)) 

    m, b, r, pval, stderr, intercept_stderr = linregress(regression_x, regression_y) 

    return regression_x, regression_y, (m, b), (r, pval, stderr, intercept_stderr) 
    
def feature_importances(model, X_test, y_test, feature_names:list, kind:str='kernel-shap'):
    
    import shap
    
    importances_df = None
    shap_values = None
    feature_importances_arr = None 

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

    feature_importances_arr = feature_importances_arr[sort_idx]
    feature_names = feature_names[sort_idx]

    if kind=='tree-shap' or kind=='kernel-shap': 
        importances_df = pd.DataFrame(shap_values, columns=feature_names[sort_idx])

    return feature_importances_arr, feature_names, importances_df
    
def confusion_matrix(y_test:numpy.array, predictions:numpy.array): 
    from sklearn.metrics import confusion_matrix, accuracy_score

    y_test = np.array(y_test).flatten()
    predictions = np.array(predictions).flatten()  

    cm = confusion_matrix(y_test, predictions)
    acc = accuracy_score(y_test, predictions)

    return cm, acc

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