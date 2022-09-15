from qpoml import collection 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression as logistic
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

qpo_path = '/mnt/c/Users/Research/Documents/GitHub/MAXI-J1535/final-push/data/pipeline/classification/MAXI_J1535-571_QPO-Input.csv'
scalar_context_path = '/mnt/c/Users/Research/Documents/GitHub/MAXI-J1535/final-push/data/pipeline/classification/MAXI_J1535-571_Scalar-Input.csv'

context_preprocess_dict = {'A':'normalize', 'B':'normalize', 'C':'normalize', 'D':'normalize', 'E':'normalize', 'F':'normalize', 'G':'normalize'}
units = {'frequency':'Hz'}

collec_one = collection()
collec_one.load(qpo_csv=qpo_path, context_csv=scalar_context_path, context_preprocess=context_preprocess_dict, approach='classification', units=units)
collec_one.evaluate(model=RandomForestClassifier(), evaluation_approach='k-fold', folds=10, stratify=True)

rf_f1 = collec_one.get_performance_statistics()['f1']

collec_two = collection()
collec_two.load(qpo_csv=qpo_path, context_csv=scalar_context_path, context_preprocess=context_preprocess_dict, approach='classification', units=units)
collec_two.evaluate(model=logistic(), evaluation_approach='k-fold', folds=10, stratify=True)

log_f1 = collec_two.get_performance_statistics()['f1']

from qpoml.plotting import plot_model_comparison
plt.style.use('/mnt/c/Users/Research/Documents/GitHub/QPOML/qpoml/stylish-2.mplstyle')
fig, ax = plt.subplots()

ax = plot_model_comparison(['Random Forest', 'Logistic Regression'], [rf_f1, log_f1], style='violin', ax=ax)
ax.set(ylabel='F1 Score')

fig.tight_layout()

plt.savefig('/mnt/c/Users/Research/Documents/GitHub/QPOML/development/testing/MAXI_J1535-571/temp_class_model_comp.png', dpi=250)

fig, ax = plt.subplots()

collec_one.plot_feature_importances(RandomForestClassifier(), fold=0, style='box', ax=ax)

plt.savefig('/mnt/c/Users/Research/Documents/GitHub/QPOML/development/testing/MAXI_J1535-571/class_feature_imps.png', dpi=150)