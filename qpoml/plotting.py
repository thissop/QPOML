import numpy
import pandas 
import warnings
import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from matplotlib.colors import LinearSegmentedColormap
#plt.style.use('https://gist.githubusercontent.com/thissop/44b6f15f8f65533e3908c2d2cdf1c362/raw/fab353d758a3f7b8ed11891e27ae4492a3c1b559/science.mplstyle')

#sns.set_style('ticks')
#sns.set_context("notebook", font_scale=0.9, rc={"font.family": 'serif'})

#plt.style.use('seaborn-darkgrid')
#plt.rcParams['font.family'] = 'serif'

sns.set_style('darkgrid')
sns.set_context("notebook") #font_scale=
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

    elif matrix_style=='with-dendrogram':
        ax = sns.clustermap(corr, center=0, cmap=bi_cm,
                   dendrogram_ratio=(.1, .2),
                   linewidths=.75, cbar_pos=(1, .2, .03, .4), 
                   row_cluster=False, metric='correlation')

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
    ax.set(xlabel='Pairwise Correlation')
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

def plot_results_regression(y_test, predictions, feature_name:str, which:list, ax=None, upper_lim_factor:float=1.025, 
                            regression_x:numpy.array=None, regression_y:numpy.array=None):
   
    from qpoml.utilities import results_regression 
    
    if regression_x is None and regression_y is None: 
        regression_x, regression_y, linregress_result = results_regression(y_test=y_test, predictions=predictions, which=which)
    
    else: 
        _, _, linregress_result = results_regression(regression_x=regression_x, regression_y=regression_y, which=None, y_test=None, predictions=None, preprocess1d_tuple=None)
    
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
    
    limits = [-0.1, upper_lim_factor*np.max(np.concatenate((regression_x, regression_y)))]
    
    ax.plot(np.array(limits), m*np.array(limits)+b, label=label) # math in equations! set that globally! 
    ax.axline(xy1=(0,0), slope=1)
    ax.set(xlim=limits, ylim=limits, xlabel='True '+feature_name, ylabel='Predicted '+feature_name)
    ax.legend()

    if internal: 
        plt.tight_layout()
        
    return ax 

def plot_feature_importances(model, X_test, y_test, feature_names:list, kind:str='kernel-shap', style:str='bar', ax=None, cut:float=2, sigma:float=2, fold=None):
    r'''
    
    Arguments
    ---------

    style : str
        Depending on the type of feature importances calculated, it can be different plot. If bar, then mean feature importances will be plotted as bar chart. However, if kind is not default, then style can also be 'box', 'violin', 'errorbar'. 
    
    cut : float
        only applicable if style is 'violin' ... from seaborn docs: 'Distance, in units of bandwidth size, to extend the density past the extreme datapoints. Set to 0 to limit the violin range within the range of the observed data (i.e., to have the same effect as trim=True in ggplot.'
    
    sigma : float 
        only applicable if style is 'errorbar' ... sigma used in calculating errorbars 
    
    '''
    
    from qpoml.utilities import feature_importances

    if ax is None: 
        fig, ax = plt.subplots()
   
    mean_importances_df, importances_df  = feature_importances(model=model, X_test=X_test, y_test=y_test, feature_names=feature_names, kind=kind, fold=fold)
    
    if kind=='default' and style!='bar':
        raise Exception('')

    if kind=='default' or style=='bar': 
        sns.barplot(data=mean_importances_df, ax=ax, color=seaborn_colors[0])

    elif importances_df is not None: 
        if style == 'violin':
            sns.violinplot(data=importances_df, ax=ax, cut=cut, color=seaborn_colors[0])
        elif style == 'box':
            sns.boxplot(data=importances_df, ax=ax, color=seaborn_colors[0]) # 
        elif style == 'errorbar':
            thickness = plt.rcParams['axes.linewidth']

            cols = list(importances_df)
            ax.errorbar(x=cols,
                        y=[np.mean(importances_df[i]) for i in cols], 
                        yerr=[sigma*np.std(importances_df[i]) for i in cols], 
                        lw=0, elinewidth=thickness, capsize=3*thickness, marker='o', 
                        color=seaborn_colors[0])

        else: 
            raise Exception('')

    else: 
        ax.boxplot(importances_df)
        ax.set_xticklabels(list(importances_df)) 

    ax.set(ylabel='Feature Name', xlabel='Feature Importance')

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

# External Utilities # 

def plot_model_comparison(model_names:list, performance_lists:list, metric:str, style:str='box', ax=None, cut:float=2, sigma:float=2):
    r'''
    Arguments
    ---------

    metric : str
        Name of the metric used to evaluate the models. 

    style : str
        'box', 'violin', or 'errorbar' 
    
    cut : float
        only applicable if style is 'violin' ... from seaborn docs: 'Distance, in units of bandwidth size, to extend the density past the extreme datapoints. Set to 0 to limit the violin range within the range of the observed data (i.e., to have the same effect as trim=True in ggplot.'
    
    sigma : float 
        only applicable if style is 'errorbar' ... sigma used in calculating errorbars 
    
    '''

    if ax is None: 
        fig, ax = plt.subplots()

    df = pd.DataFrame(list(zip(performance_lists)), columns=model_names)

    if style == 'violin':
        sns.violinplot(data=df, ax=ax, cut=cut, color=seaborn_colors[0])
    elif style == 'box':
        sns.boxplot(data=df, ax=ax, color=seaborn_colors[0]) # 
    elif style == 'errorbar':
        thickness = plt.rcParams['axes.linewidth']

        cols = list(df)
        ax.errorbar(x=cols,
                    y=[np.mean(df[i]) for i in cols], 
                    yerr=[sigma*np.std(df[i]) for i in cols], 
                    lw=0, elinewidth=thickness, capsize=3*thickness, marker='o', 
                    color=seaborn_colors[0])

    else: 
        raise Exception('')

    ax.set(ylabel=metric)

    return ax