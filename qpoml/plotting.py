import os
import numpy
import pandas 
import warnings
import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
from matplotlib.colors import LinearSegmentedColormap

wh1 = False
if os.path.exists('/ar1/PROJ/fjuhsd/personal/thaddaeus/github/QPOML/qpoml/stylish.mplstyle'):
    wh1 = True 

if wh1: 
    plt.style.use('/ar1/PROJ/fjuhsd/personal/thaddaeus/github/QPOML/qpoml/stylish.mplstyle')
else: 
    plt.style.use('/mnt/c/Users/Research/Documents/GitHub/QPOML/qpoml/stylish.mplstyle')

#plt.style.use('/mnt/c/Users/Research/Documents/GitHub/QPOML/qpoml/stylish-2.mplstyle')
#sns.set_style('ticks')
#sns.set_context("notebook", font_scale=0.9, rc={"font.family": 'serif'})

#plt.style.use('seaborn-darkgrid')
#plt.rcParams['font.family'] = 'serif'
#sns.set_style('darkgrid')
#plt.style.use('https://gist.githubusercontent.com/thissop/44b6f15f8f65533e3908c2d2cdf1c362/raw/fab353d758a3f7b8ed11891e27ae4492a3c1b559/science.mplstyle')
sns.set_context("paper") #font_scale=
sns.set_palette('deep')
seaborn_colors = sns.color_palette('deep')

plt.rcParams['font.family'] = 'serif'
plt.rcParams["mathtext.fontset"] = "dejavuserif"

bi_cm = LinearSegmentedColormap.from_list("Custom", [seaborn_colors[0], (1,1,1), seaborn_colors[3]], N=20)
bi_cm_r = LinearSegmentedColormap.from_list("Custom", [seaborn_colors[3], (1,1,1), seaborn_colors[0]], N=20)

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

    fig, ax = plt.subplots(figsize=(4,2))
    x = np.arange(0,len(scores),1)

    ax.plot(x, np.array(scores))
    ax.fill_between(x,scores-stds, scores+stds, alpha=0.1, color='cornflowerblue')
    ax.xaxis.set_ticklabels([])
    ax.set(ylabel='Score')

    return ax 

### POST EVALUATION ### 

def plot_results_regression(y_test, predictions, feature_name:str, which:list, ax=None, upper_lim_factor:float=1.025, 
                            regression_x:numpy.array=None, regression_y:numpy.array=None, unit:str=None, font_scale:float=1.15):
   
    from qpoml.utilities import results_regression 
    
    sns.set_context(font_scale=font_scale)

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

    ax.scatter(regression_x, regression_y, edgecolors='black', lw=0.5)
    r_sq = str(round(r**2, 2))
    label = r'$r^2=$'+' '+r_sq+'\n'+r'$\frac{\Delta y}{\Delta x}=$'+str(round(m, 2))+r'$\pm$'+stderr 
    
    limits = [-0.1, upper_lim_factor*np.max(np.concatenate((regression_x, regression_y)))]
    
    ax.plot(np.array(limits), m*np.array(limits)+b, label=label, color='black')# color=seaborn_colors[7]) # math in equations! set that globally! 
    ax.axline(xy1=(0,0), slope=1, ls='--', color='black')#color=seaborn_colors[7]
    
    label_suffix = feature_name.title() 
    if unit is not None: 
        label_suffix += f' ({unit})'

    ax.set(xlim=limits, ylim=limits, xlabel=f'True {label_suffix}', ylabel=f'Predicted {label_suffix}')
    ax.legend(loc='upper left')

    if internal: 
        plt.tight_layout()
        
    return ax 

def plot_feature_importances(model, X_test, y_test, feature_names:list, kind:str='tree-shap', 
                             style:str='bar', ax=None, cut:float=2, sigma:float=2.576, 
                             mean_importances_df:pandas.DataFrame=None, importances_df:pandas.DataFrame=None, confidence_level:float=0.99, hline:bool=False):
    r'''
    
    Arguments
    ---------

    style : str
        Depending on the type of feature importances calculated, it can be different plot. If bar, then mean feature importances will be plotted as bar chart. However, if kind is not default, then style can also be 'box', 'violin', 'errorbar'. 
    
    cut : float
        only applicable if style is 'violin' ... from seaborn docs: 'Distance, in units of bandwidth size, to extend the density past the extreme datapoints. Set to 0 to limit the violin range within the range of the observed data (i.e., to have the same effect as trim=True in ggplot.'
    
    sigma : float 
        only applicable if style is 'errorbar' ... sigma used in calculating errorbars 

    hline : bool 
        if it's true, median feature importance will be plotted as dashed line, and mean feature importance as dotted line
    
    '''
    
    from qpoml.utilities import feature_importances
    import scipy 

    mpl.rcParams.update(mpl.rcParamsDefault)
    if wh1: 
        plt.style.use('/ar1/PROJ/fjuhsd/personal/thaddaeus/github/QPOML/qpoml/stylish.mplstyle')
    else: 
        plt.style.use('/mnt/c/Users/Research/Documents/GitHub/QPOML/qpoml/stylish.mplstyle')
    sns.set_context('paper')

    #sns.set_context('paper')

    if ax is None: 
        fig, ax = plt.subplots()
    
    if mean_importances_df is None or importances_df is None: 
        mean_importances_df, importances_df  = feature_importances(model=model, X_test=X_test, y_test=y_test, feature_names=feature_names, kind=kind)
    
    if kind=='default' and style!='bar':
        raise Exception('')

    if kind=='default' or style=='bar': 
        yerr = np.array([2.576*np.std(importances_df[i])/np.sqrt(len(importances_df[i])) for i in list(importances_df)]) # 99% CI 
        sns.barplot(data=mean_importances_df, ax=ax, color=seaborn_colors[0], edgecolor='black', linewidth=0.7, yerr=yerr)

        if hline is not None and hline: 
            ax.axhline(y=np.median(mean_importances_df), ls='--', color='black')
            ax.axhline(y=np.mean(mean_importances_df, axis=1)[0], ls=':', color='black')

        '''
        
        standard error notes
        
        - standard error shows the accuracy of the mean...estimates standard deviation the mean's sampling distribution (SEM = standard error of the mean)
        

        - error bars based on the s.e.m. reflect the uncertainty in the mean and its dependency on the sample size

        - CI: This is an interval estimate that indicates the reliability of a measurement
        '''

    elif importances_df is not None: 
        if style == 'violin':
            sns.violinplot(data=importances_df, ax=ax, cut=cut, color=seaborn_colors[0])
        elif style == 'box':
            sns.boxplot(data=importances_df, ax=ax, color=seaborn_colors[0], whis=2.) # 
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
        ax.boxplot(importances_df, whis=2.0)
        ax.set_xticklabels(list(importances_df)) 

    ax.set(xlabel='Feature', ylabel='Feature Importance')

    return ax

#### CLASSIFICATION ####

def plot_confusion_matrix(y_test:numpy.array, predictions:numpy.array, auc:float=None, ax=None, cbar:bool=False, labels:list=None): 
    from qpoml.utilities import confusion_matrix 

    mpl.rcParams.update(mpl.rcParamsDefault)
    if wh1: 
        plt.style.use('/ar1/PROJ/fjuhsd/personal/thaddaeus/github/QPOML/qpoml/stylish.mplstyle')
    else: 
        plt.style.use('/mnt/c/Users/Research/Documents/GitHub/QPOML/qpoml/stylish.mplstyle')
    sns.set_context('paper', font_scale=1.4)

    internal = False 
    if ax is None: 
        fig, ax = plt.subplots()
        internal = True

    cm, acc = confusion_matrix(y_test=np.array(y_test), predictions=np.array(predictions))

    if labels is None: 
        sns.heatmap(cm, annot=True, linewidths=.5, ax=ax, center=0.0, cmap=bi_cm_r, cbar=cbar, annot_kws={'fontsize':'xx-large'})
    else: 
        sns.heatmap(cm, annot=True, linewidths=.5, ax=ax, center=0.0, cmap=bi_cm_r, cbar=cbar, yticklabels=labels, xticklabels=labels, annot_kws={'fontsize':'xx-large'})
    
    ax.set(xlabel='True Class', ylabel='Predicted Class')#, xticks=[0,1], yticks=[0,1])
    #ax.axis('off')
    #ax.tick_params(top=False, bottom=False, left=False, right=False)

    ax.patch.set_edgecolor('black')  

    ax.patch.set_linewidth('3')

    if internal: 
        fig.tight_layout()

    return ax

def plot_roc(fpr:np.array, tpr:np.array, std_tpr:float=None, ax=None, auc:float=None, std_auc:float=None):
    '''
    
    Notes
    -----

    - Portions of this routine were modified from an sklearn documentation example that can be found at this [link](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html?highlight=roc+curve)

    '''

    mpl.rcParams.update(mpl.rcParamsDefault)
    if wh1: 
        plt.style.use('/ar1/PROJ/fjuhsd/personal/thaddaeus/github/QPOML/qpoml/stylish.mplstyle')
    else: 
        plt.style.use('/mnt/c/Users/Research/Documents/GitHub/QPOML/qpoml/stylish.mplstyle')
    sns.set_context('paper')

    if ax is None: 
        fig, ax = plt.subplots()

    if std_auc is not None: 
        roc_label = r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (auc, std_auc)
    else: 
        roc_label = 'ROC (AUC = %0.2f )' % auc

    ax.plot(fpr,tpr, label=roc_label, lw=1, alpha=1)
    ax.plot([0, 1], [0, 1], linestyle="--", lw=1, color='black', label="Chance", alpha=1)

    if std_tpr is not None: 
        tprs_upper = np.minimum(tpr + std_tpr, 1)
        tprs_lower = np.maximum(tpr - std_tpr, 0)
        ax.fill_between(
            fpr,
            tprs_lower,
            tprs_upper,
            color="grey",
            alpha=0.2)

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], xlabel='False Positive Rate', ylabel='True Positive Rate')
    if auc is not None: 
        ax.text(0.6, 0.2, f'AUC={round(auc,2)}', transform = ax.transAxes, size='large')

    #ax.legend(loc="lower right", fontsize='xx-small')

    return ax  
    
# External Utilities # 

def bias_report(predictions, y_test, feature_names:list, ax:None, fold:int=0):
    r'''
    
    Arguments
    ---------

    predictions 
        - list of shape (m, n) where (m) is number of 'rows' or instances/observations, and (n) is number of features. Can be for multiple folds, zeroth will be selected for study by default

    y_test 
        - same, but for true values

     
    '''

    from scipy import stats 

    feature_names = feature_names*int(len(predictions)/len(feature_names))

    predictions, y_test = (np.transpose(i) for i in [predictions[fold], y_test[fold]])

    diffs = [i-j for i, j in zip(predictions, y_test)]

    sns.set_context("paper", font_scale=0.8) 

    df = pd.DataFrame()

    for i in range(len(diffs)): 
        df[feature_names[i]] = diffs[i]
    
    if ax is None: 
        fig, ax = plt.subplots()
    
    sns.boxplot(data=df, color=seaborn_colors[0])
    ax.set(xlabel='QPO Parameter', ylabel='Normalized Residual')

    ks_ps = {} 

    for i in range(len(diffs)):
        for j in range(i, len(diffs)-1):
            p = stats.kstest(diffs[i], diffs[j], alternative='two-sided')[1]
            ks_ps.update({f'{feature_names[i]}-{feature_names[j]}':[p]})

    ks_df = pd.DataFrame(ks_ps)

    return ks_df 

def plot_model_comparison(model_names:list, performance_lists:list, style:str='box', ax=None, fig=None, ylabel='Median Absolute Error', cut:float=2, sigma:float=2):
    r'''
    Arguments
    ---------

    metric : str
        Name of the metric used to evaluate the models. 

    style : str
        'box', 'violin', or 'errorbar' 
    
    cut : float
        only applicable if style is 'violin' ... from seaborn docs: 'Distance, in units of bandwidth size, to extend the density past the extreme datapoints. Set to 0 to limit the violin range within the range of the observed data (i.e., to have the same effect as trim=True in ggplot.'
    
        **NOTE** this could be fixed by incorporating clip too 

    sigma : float 
        only applicable if style is 'errorbar' ... sigma used in calculating errorbars 
    
    '''

    #mpl.rcParams.update(mpl.rcParamsDefault)
    #sns.set_style('whitegrid')
    #plt.rcParams['font.family']='serif'
    #plt.rcParams['xtick.minor.visible'] = False
    #ax.patch.set_edgecolor('black')  
    #ax.patch.set_linewidth('1')

    #plt.style.use('/mnt/c/Users/Research/Documents/GitHub/QPOML/qpoml/stylish-2.mplstyle')

    sort_idx = np.argsort([np.mean(i) for i in performance_lists])[::-1]

    performance_lists, model_names = (np.array(performance_lists)[sort_idx], np.array(model_names)[sort_idx])

    sns.set_context(font_scale=1.3)

    if ax is None: 
        fig, ax = plt.subplots()
    
    if fig is not None: 
        fig.supxlabel('Model', fontsize='small')

    df = pd.DataFrame()
    for i in range(len(performance_lists)):
        df[model_names[i]] = performance_lists[i]

    if style == 'violin':
        sns.violinplot(data=df, ax=ax, cut=cut, color=seaborn_colors[0])
        #ax.set_xticks([], which='minor')
        ax.set_xticks(np.arange(0, len(list(df))), which='major')
    elif style == 'box':
        sns.boxplot(data=df, ax=ax, color=seaborn_colors[0], whis=2.0) # 
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

    ax.set_xticklabels(labels=ax.get_xticklabels(), fontsize='x-small')
    ax.set_ylabel(ylabel, fontsize='small')
    

    
    

    return ax