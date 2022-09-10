from qpoml import collection 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
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

print(collec.predictions[0])
print(collec.y_test[0])

from qpoml.plotting import plot_roc  
import os 
os.chdir('/mnt/c/Users/Research/Documents/GitHub/QPOML/development/testing/MAXI_J1535-571')

fig, ax = plt.subplots(figsize=(4,4))

fpr, tpr, std_tpr, auc, std_auc = collec.roc_and_auc()

plot_roc(fpr, tpr, auc, std_auc, std_tpr, ax = ax)

plt.savefig('temp_roc.png', dpi=150)

plt.clf()
plt.close()

fig, ax = plt.subplots(figsize=(4,4))

collec.plot_confusion_matrix(ax=ax)

plt.savefig('temp_matrix.png', dpi=150)

plt.clf()
plt.close()