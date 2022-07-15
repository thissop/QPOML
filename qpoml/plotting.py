import numpy
import pandas 
import warnings
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 
from matplotlib.colors import LinearSegmentedColormap
#plt.style.use('https://gist.githubusercontent.com/thissop/44b6f15f8f65533e3908c2d2cdf1c362/raw/fab353d758a3f7b8ed11891e27ae4492a3c1b559/science.mplstyle')

#sns.set_style('ticks')
#sns.set_context("notebook", font_scale=0.9, rc={"font.family": 'serif'})

#plt.style.use('seaborn-darkgrid')
#plt.rcParams['font.family'] = 'serif'

sns.set_style('darkgrid')
sns.set_context("paper") #font_scale=
sns.set_palette('deep')

seaborn_colors = sns.color_palette('deep')

bi_cm = LinearSegmentedColormap.from_list("Custom", [seaborn_colors[0], (1,1,1), seaborn_colors[3]], N=20)

### POST LOAD ### 

def plot_correlation_matrix(data:pandas.DataFrame, ax=None, matrix_style:str='default'):
    from qpoml.utilities import correlation_matrix 

    corr, cols = correlation_matrix(data=data)

    internal = False 

    if ax is None: 
        fig, ax = plt.subplots()

    if matrix_style=='default': 
            sns.heatmap(corr,
                ax=ax, annot=True, annot_kws={'fontsize':'small'}, yticklabels=cols,
                xticklabels=cols, cbar_kws={"shrink": .75}, center=0.0, cmap=bi_cm)

    elif matrix_style=='steps': 
        mask = np.triu(np.ones_like(corr, dtype=bool))   
        sns.heatmap(corr, mask=mask, ax=ax, annot=True, annot_kws={'fontsize':'small'},
                    yticklabels=cols, xticklabels=cols, cbar_kws={"shrink": .75}, center=0.0, 
                    cmap=bi_cm)

    else: 
        raise Exception('')

    return ax

def plot_pairplot(data:pandas.DataFrame, steps=False, ax=None): 

    if ax is None: 
        fig, ax = plt.subplots()
        internal = True 

    ax = sns.pairplot(data=data, corner=steps)
    
    return ax

def plot_dendrogram(data:pandas.DataFrame, ax=None): 
    from scipy.cluster import hierarchy
    from qpoml.utilities import dendrogram

    corr, dist_linkage, cols = dendrogram(data=data)
    
    internal = False 
    if ax is None: 
        fig, ax = plt.subplots()
        internal = True 

    hierarchy.dendrogram(dist_linkage, labels=cols, ax=ax, leaf_rotation=90)

    if internal: 
        plt.tight_layout()

    return ax

def plot_vif(data:pandas.DataFrame, cutoff:int=10, ax=None): 
    from qpoml.utilities import calculate_vif 

    internal = False 
    if ax is None: 
        fig, ax = plt.subplots()
        internal = True 

    vif_df = calculate_vif(data=data)

    ax.barh(vif_df['Column'], vif_df['VIF'])
    ax.axvline(x=cutoff, ls='--')
    ax.set(ylabel='Feature Name', xlabel='Variance Inflation Factor (VIF)')

    if internal: 
        plt.tight_layout()

    return ax

def plot_gridsearch(scores, ax=None):     
    if ax is None: 
        fig, ax = plt.subplots(figsize=(4,2))

    scores = np.array(scores)
    stds = np.array(stds)

    # grid search results plot 
    sns.set_style('darkgrid')
    sns.set_context("paper", font_scale=0.5) #font_scale=
    sns.set_palette('deep')

    fig, ax = plt.subplots(figsize=(4,2))
    x = np.arange(0,len(scores),1)

    ax.plot(x, np.array(scores))
    ax.fill_between(x,scores-stds, scores+stds, alpha=0.1, color='cornflowerblue')
    ax.xaxis.set_ticklabels([])
    ax.set(ylabel='Score')

    return ax 

### POST EVALUATION ### 

def plot_results_regression(y_test, predictions, feature_name:str, which:list, ax=None, xlim:list=[-0.1,1.1], 
                            regression_x:numpy.array=None, regression_y:numpy.array=None):
   
    from qpoml.utilities import results_regression 
    
    if regression_x is None and regression_y is None: 
        regression_x, regression_y, linregress_result = results_regression(y_test=y_test, predictions=predictions, which=which)
    else: 
        _, _, linregress_result = results_regression(regression_x=regression_x, regression_y=regression_y, which=None, y_test=None, predictions=None)
    
    internal = False 
    if ax is None: 
        fig, ax = plt.subplots() 
        internal = True 
    
    m = linregress_result[0]
    b = linregress_result[1]
    r = linregress_result[2]
    stderr = str(round(linregress_result[4], 2))

    ax.scatter(regression_x, regression_y)
    r_sq = str(round(r**2, 2))
    label = 'Best Fit ('r'$r^2=$'+' '+r_sq+', m='+str(round(m, 2))+r'$\pm$'+stderr 
    ax.plot(np.array(xlim), m*np.array(xlim)+b, label=label) # math in equations! set that globally! 
    ax.axline(xy1=(0,0), slope=1)
    ax.set(xlim=xlim, ylim=xlim, xlabel='True '+feature_name, ylabel='Predicted '+feature_name)
    ax.legend()

    if internal: 
        plt.tight_layout()
        
    return ax 

def plot_feature_importances(model, X_test, y_test, feature_names:list, kind:str='kernel-shap', ax=None):
    from qpoml.utilities import feature_importances

    internal = False 
    if ax is None: 
        fig, ax = plt.subplots()
        internal = True 
    
    feature_importances_arr, _, importances_df = feature_importances(model=model, X_test=X_test, y_test=y_test, feature_names=feature_names, kind=kind)
    if importances_df is None or kind in ['tree-shap', 'kernel-shap']: # fix this! return errs! fix that in calculate feature importances! 
        ax.barh(feature_names, feature_importances_arr)
        ax.set(xlabel='Feature Importance')

        warnings.warn('need to set ')

    else: 
        ax.boxplot(importances_df)
        ax.set_xticklabels(list(importances_df)) 

    ax.set(ylabel='Feature Name')

    if internal: 
        plt.tight_layout()

    return ax

def plot_confusion_matrix(y_test:numpy.array, predictions:numpy.array, ax=None): 
    from qpoml.utilities import confusion_matrix 

    internal = False 
    if ax is None: 
        fig, ax = plt.subplots()
        internal = True

    cm, acc = confusion_matrix(y_test=np.array(y_test), predictions=np.array(predictions))

    sns.heatmap(cm, annot=True, center=0.0, linewidths=.5, ax=ax)
    ax.set(xlabel='Actual', ylabel='Predicted')

    warnings.warn('replace cmap!')

    if internal: 
        plt.tight_layout()

    return ax