from qpoml import collection 
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

qpo_path = '/mnt/c/Users/Research/Documents/GitHub/MAXI-J1535/final-push/data/pipeline/regression/GRS_1915+105_QPO-Input.csv'
scalar_context_path = '/mnt/c/Users/Research/Documents/GitHub/MAXI-J1535/final-push/data/pipeline/regression/GRS_1915+105_Scalar-Input.csv'

context_preprocess_dict = {'A':'normalize', 'B':'normalize', 'C':'normalize', 'D':'normalize', 'E':'normalize', 'F':'normalize', 'G':'normalize'}
units = {'frequency':'Hz'}
qpo_preprocess = {'frequency':'normalize', 'width':'normalize', 'rms':'normalize'}

collec_one = collection()
collec_one.load(qpo_csv=qpo_path, context_csv=scalar_context_path, context_preprocess=context_preprocess_dict, qpo_preprocess=qpo_preprocess, approach='regression', units=units)
collec_one.evaluate(model=RandomForestRegressor(), evaluation_approach='k-fold', folds=10, stratify=True)

rf_mae = collec_one.get_performance_statistics()['mae']

collec_two = collection()
collec_two.load(qpo_csv=qpo_path, context_csv=scalar_context_path, context_preprocess=context_preprocess_dict, qpo_preprocess=qpo_preprocess, approach='regression', units=units)
collec_two.evaluate(model=LinearRegression(), evaluation_approach='k-fold', folds=10, stratify=True)

ols_mae = collec_two.get_performance_statistics()['mae']

plt.style.use('/mnt/c/Users/Research/Documents/GitHub/QPOML/qpoml/stylish.mplstyle')

fig, axs = plt.subplots(1, 2, figsize=(8,4))

ax = axs[0]

collec_two.plot_results_regression('frequency', which=[0], ax=ax)

ax = axs[1]

collec_one.plot_results_regression('frequency', which=[0], ax=ax)

plt.savefig('/mnt/c/Users/Research/Documents/GitHub/QPOML/development/testing/GRS_1915+105/temp_reg_model_comp.png', dpi=250)

from qpoml.utilities import compare_models

tstat, pval = compare_models(rf_mae, ols_mae, n_train=499, n_test=55, approach='frequentist', better='lower')

print(tstat, pval)

tstat, pval = compare_models(ols_mae, 10*np.array(ols_mae), n_train=499, n_test=55, approach='frequentist', better='lower')

print(tstat, pval)

tstat, pval = compare_models(10*np.array(ols_mae), ols_mae, n_train=499, n_test=55, approach='frequentist', better='lower')

print(tstat, pval)

# fig five results # 

predictions = np.transpose(collec_one.predictions[0])
y_test = np.transpose(collec_one.y_test[0])

observation_IDs = np.array(collec_one.observation_IDs)[collec_one.test_indices[0]]

freqs_pred = predictions[0]
freqs_true = y_test[0]
widths_pred = predictions[1]
widths_true = y_test[1]
amps_pred = predictions[2]
amps_true = y_test[2]

freq_preprocess1d_tuple = collec_one.qpo_preprocess1d_tuples['frequency']
width_preprocess1d_tuple = collec_one.qpo_preprocess1d_tuples['width']
rms_preprocess1d_tuple = collec_one.qpo_preprocess1d_tuples['rms']

from qpoml.utilities import preprocess1d, unprocess1d

freqs_pred = unprocess1d(freqs_pred, freq_preprocess1d_tuple) 
freqs_true = unprocess1d(freqs_true, freq_preprocess1d_tuple) 
widths_pred = unprocess1d(widths_pred, width_preprocess1d_tuple) 
widths_true = unprocess1d(widths_true, width_preprocess1d_tuple) 
amps_pred =  unprocess1d(amps_pred, rms_preprocess1d_tuple)
amps_true = unprocess1d(amps_true, rms_preprocess1d_tuple)

aes = np.abs(freqs_pred-freqs_true) # absolute errors 

temp_df = pd.DataFrame()
for col_name, col in zip(['observation_ID', 'ae', 'freq_pred', 'freq_true', 'width_pred', 'width_true', 'rms_true', 'rms_pred'], [observation_IDs, aes, freqs_pred,freqs_true,widths_pred,widths_true,amps_pred,amps_true]):
    temp_df[col_name] = col

temp_df = temp_df.sort_values(by='ae')

temp_df.to_csv('/mnt/c/Users/Research/Documents/GitHub/QPOML/development/testing/GRS_1915+105/fig_five_info.csv', index=False)