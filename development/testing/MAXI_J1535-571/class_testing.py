from qpoml import collection 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression as logistic
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

qpo_path = '/mnt/c/Users/Research/Documents/GitHub/MAXI-J1535/final-push/data/pipeline/classification/MAXI_J1535-571_QPO-Input.csv'
scalar_context_path = '/mnt/c/Users/Research/Documents/GitHub/MAXI-J1535/final-push/data/pipeline/classification/MAXI_J1535-571_Scalar-Input.csv'

context_preprocess_dict = {'A':'as-is', 'B':'as-is', 'C':'as-is', 'D':'as-is', 'E':'as-is', 'F':'as-is', 'G':'as-is'}
units = {'frequency':'Hz'}

collec = collection()
collec.load(qpo_csv=qpo_path, context_csv=scalar_context_path, context_preprocess=context_preprocess_dict, approach='classification', units=units)
collec.evaluate(model=RandomForestClassifier(), evaluation_approach='k-fold', folds=10, stratify=True)

import seaborn as sns 
import matplotlib.pyplot as plt 
from matplotlib.colors import LinearSegmentedColormap

#plt.rcParams['font.family'] = 'serif'
#plt.style.use('https://gist.githubusercontent.com/thissop/44b6f15f8f65533e3908c2d2cdf1c362/raw/fab353d758a3f7b8ed11891e27ae4492a3c1b559/science.mplstyle')
#sns.set_context("paper") #font_scale=
#sns.set_palette('deep')
#seaborn_colors = sns.color_palette('deep')

#plt.rcParams['font.family'] = 'serif'
#plt.rcParams["mathtext.fontset"] = "dejavuserif"

#bi_cm = LinearSegmentedColormap.from_list("Custom", [seaborn_colors[0], (1,1,1), seaborn_colors[3]], N=20)

from qpoml.plotting import plot_roc  
import os 

os.chdir('/mnt/c/Users/Research/Documents/GitHub/QPOML/development/testing/MAXI_J1535-571')

fig, ax = plt.subplots(figsize=(4,4))

fpr, tpr, std_tpr, auc, std_auc = collec.roc_and_auc()

plot_roc(fpr, tpr, std_tpr, ax = ax, auc=std_auc)

plt.savefig('temp_roc.png', dpi=150)

plt.clf()
plt.close()

fig, ax = plt.subplots(figsize=(4,4))

ax = collec.plot_confusion_matrix(ax=ax, labels=['QPO', 'No QPO'])

plt.savefig('temp_matrix.png', dpi=150)

plt.clf()
plt.close()