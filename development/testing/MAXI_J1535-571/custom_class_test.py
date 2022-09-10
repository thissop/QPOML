from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression as logistic
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

scalar_context_path = '/mnt/c/Users/Research/Documents/GitHub/MAXI-J1535/final-push/data/pipeline/classification/input.csv'
qpo_path = '/mnt/c/Users/Research/Documents/GitHub/MAXI-J1535/final-push/data/pipeline/classification/output.csv'
qpo_path = '/mnt/c/Users/Research/Documents/GitHub/MAXI-J1535/final-push/data/pipeline/classification/MAXI_J1535-571_QPO-Input.csv'
scalar_context_path = '/mnt/c/Users/Research/Documents/GitHub/MAXI-J1535/final-push/data/pipeline/classification/MAXI_J1535-571_Scalar-Input.csv'

input_df = pd.read_csv(scalar_context_path)
qpo_df = pd.read_csv(qpo_path)

merged_df = input_df.merge(qpo_df, on='observation_ID').sample(frac=1)

qpo_states = np.array(merged_df['qpo_state'])

X = np.array(merged_df.drop(columns=['observation_ID', 'qpo_state']))

X_train, X_test, y_train, y_test = train_test_split(X, qpo_states, test_size=0.25, stratify=qpo_states)

print(X_train[0])

clf = SVC()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

predictions = clf.predict(X_test)

from qpoml.plotting import plot_confusion_matrix, plot_roc
from qpoml.utilities import roc_and_auc
import os 

os.chdir('/mnt/c/Users/Research/Documents/GitHub/QPOML/development/testing/MAXI_J1535-571')

fig, ax = plt.subplots(figsize=(4,4))

plot_confusion_matrix(y_test, predictions)

plt.savefig('custom_confusion_matrix.png', dpi=200)
plt.clf()
plt.close()

print(predictions, y_test)

fig, ax = plt.subplots(figsize=(4,4))

fpr, tpr, auc_score = roc_and_auc(y_test, predictions)

plot_roc(fpr, tpr, auc_score, ax = ax)

plt.savefig('custom_roc.png', dpi=150)
