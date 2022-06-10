import numpy as np
import warnings

def train_test_split(input_observations, output_observations, 
                     train_size=None, random_state=None, 
                     shuffle=True, stratify=None): 
    
    from sklearn.model_selection import train_test_split as tts

    def to_numpy():
        pass
    def from_numpy(): 
        pass 

    X = np.array([])
    y = np.array([])

    X_train, X_test, y_train, y_test = tts(X, y, 
                                           train_size=None, 
                                           random_state=None, 
                                           shuffle=True, 
                                           stratify=None)

    return X_train, X_test, y_train, y_test  

def lol(x, y, x_preprocess:dict=None, y_preprocess:dict=None,
        train_size=None, random_state=None, 
        shuffle=True, stratify=None): 

    from sklearn.model_selection import train_test_split as tts
    from qpoml import qpo, observation

    # to do: 
    '''
        - split into numpy arrays for ML
        - preprocess arrays with min max scaling 
        - turn categorical values to numerical (e.g. qpo type to ordinal)
        - return for ML
        - then do pipeline 
    '''

    '''
    features_0 = np.array(list(x_0.features.values()))
    x_mask = features_0!=None
    x_length = np.sum(x_mask) 
    X = np.array([]).reshape(0, x_length)
    X = np.vstack([X, values])
    '''

    if len(x)!=len(y): 
        raise Exception('x and y must have same length')

    x_0 = x[0] 
    names_0 = np.array(list(x_0.features.keys()))
    features_0 = np.array(list(x_0.features.values()))
    x_mask = features_0!=None

    names_0, features_0 = (names_0[x_mask], features_0[x_mask])

    types = [type(i) for i in features_0]
    
    X = [np.array([], dtype=type(i)) for i in features_0]

    X = dict(zip(names_0, X))

    for item in x: 
        names, features = ()

    if isinstance(y[0], qpo): 
        pass 

    elif isinstance(y[0], observation): 
        pass 
    else: 
        raise Exception('illegal y value') 


def normalize(x, min_value:float=None, max_value:float=None): 

    if min_value==None and max_value == None: 
        min_value = np.min(x)
        max_value = np.max(x)
    
    return (x-min_value)/(max_value-min_value), (min_value, max_value)

def standardize(x, mean:float=None, sigma:float=None): 
    if mean==None and sigma==None: 
        mean = np.mean(x)
        sigma = np.std(x)
    
    return (x-mean)/sigma, (mean, sigma)

def inverse_encoder_transform(x, original_classes): 
    classes_dict = dict(zip(list(range(len(original_classes))), original_classes))
    x = np.array([classes_dict[i] for i in x])
    return x 
    
    # for best practice, should they only do standardization or normalization for all continuous variables? 

warnings.warn('include option to do standardize or normalize data')
warnings.warn('by default, float variables are normalized/standardized, whereas integer variables are not')