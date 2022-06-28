import numpy as np

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

