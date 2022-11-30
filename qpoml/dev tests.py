import keras 
from keras import layers 
from keras import models 
import sklearn 
from xgboost import XGBRegressor

xgboost_model = XGBRegressor()
keras_model = models.Sequential()
keras_model.add(layers.Dense(32, input_shape=(784,)))
keras_model.add(layers.Dense(32))

cloned_xgboost = sklearn.base.clone(xgboost_model)
cloned_keras = sklearn.base.clone(keras_model)

print('cloning successful')

cloned_xgboost = cloned_xgboost.set_params(**{'n_estimators':5})
#cloned_keras = cloned_keras.set_params(**{''})

print('hyperparameter setting successful')