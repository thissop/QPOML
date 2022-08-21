from qpoml import collection 
import numpy as np
import pandas as pd 
from sklearn.ensemble import RandomForestRegressor as RandomForest

source = 'GRS_1915+105'
model = RandomForest()
model_name = 'Random Forest'

model_hyperparameter_dictionary = {'min_samples_split':[4,6], 'min_samples_leaf':[3]}
                                     
source_classes = ['BH']
source_instruments = ['RXTE']

units = {'frequency':'Hz'}

input_directory='/mnt/c/Users/Research/Documents/GitHub/MAXI-J1535/final-push/data/pipeline/'
output_directory = '/mnt/c/Users/Research/Documents/GitHub/QPOML/development/testing/'

scalar_context_path = '/mnt/c/Users/Research/Documents/GitHub/MAXI-J1535/final-push/data/pipeline/GRS_1915+105_Scalar-Input.csv'
qpo_path = '/mnt/c/Users/Research/Documents/GitHub/MAXI-J1535/final-push/data/pipeline/GRS_1915+105_QPO-Input.csv'

qpo_preprocess_dict = {'frequency':'normalize','width':'normalize', 'rms':'normalize'}
context_preprocess_dict = {'A':'normalize', 'B':'normalize', 'C':'normalize', 'D':'normalize', 'E':'normalize', 'F':'normalize', 'G':'normalize'}

fold_performances = []
gridsearch_scores = []

# 2.1: Machine Learning for Each Model # 

model_name = 'Random Forest'

collec= collection()
collec.load(qpo_csv=qpo_path, context_csv=scalar_context_path, context_type='scalar',  
            context_preprocess=context_preprocess_dict, qpo_preprocess=qpo_preprocess_dict, units=units) 

# 2.1.1: GridSearch # 

scores, _, _, best_params = collec.gridsearch(model=model, parameters=model_hyperparameter_dictionary)

column_names = list(best_params.keys())
columns = [[i] for i in list(best_params.values())]

best_configuration_df = pd.DataFrame()

for temp_index in range(len(column_names)):
    best_configuration_df[column_names[temp_index]] = columns[temp_index]

best_configuration_df.to_csv(f'{output_directory}{source}/{model_name}_BestParams.csv', index=False)

gridsearch_scores.append(scores)

# 2.1.2: k-fold on Best Configuration # 

collec.evaluate(model=model, evaluation_approach='k-fold', folds=10, repetitions=2, hyperparameter_dictionary=best_params) # evaluate-approach???


y_test = collec.y_test[9]
predictions = collec.predictions[9]
test_observation_IDs = collec.test_observation_IDs[9]

cols = ['frequency', 'width', 'rms']
true_df = pd.DataFrame(y_test, columns=[f'true_{i}' for i in cols])
predicted_df = pd.DataFrame(predictions, columns=[f'predicted_{i}' for i in cols])

out_df = pd.DataFrame()
out_df['observation_ID'] = test_observation_IDs

frequency_quotient = true_df['true_frequency']/predicted_df['predicted_frequency']

out_df['frequency_quotient'] = frequency_quotient

out_df = out_df.join(true_df)
out_df = out_df.join(predicted_df)

out_df = out_df.sort_values(by='frequency_quotient')

print(out_df)

out_df.to_csv('/mnt/c/Users/Research/Documents/GitHub/QPOML/development/testing/matching_to_obsids/fold=9.csv', index=False)

print(collec.qpo_preprocess1d_tuples)

r'''
50703-01-67-00 looks good
'''