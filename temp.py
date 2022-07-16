import numpy as np

def preprocess1d(x, preprocess, range_low:float=0.1, range_high:float=1.0): 

    r'''
    
    range_low, range_high : float, float 
        feature range that inputs are mapped to. default is 0.1 to 1.0 

    '''

    if type(preprocess) is str: 
        
        if preprocess == 'as-is': 
            modified = x 
            return modified, ('as-is', None, None) 
        
        elif preprocess == 'normalize': 
            min_value = np.min(x)
            max_value = np.max(x)
            modified = (x-min_value)/(max_value-min_value)
            modified = modified*(range_high - range_low) + range_low
            return modified, ('normalize', min_value, max_value)
        
        elif preprocess == 'standardize': 
            mean = np.mean(x)
            sigma = np.std(x)
            modified = (x-mean)/sigma
            return modified, ('standardize', mean, sigma)

        elif preprocess == 'median': 
            median = np.median(x)
            modified = x/median
            return modified, ('median', median, None) 
        
        else: 
            raise Exception('')
    
    else: 
        try: 
            min_value = preprocess[0]
            max_value = preprocess[1]
            modified = (x-min_value)/(max_value-min_value) 
            modified = modified*(range_high - range_low) + range_low # so it will be output as 0.1-1 range 
            return modified, ('normalize', min_value, max_value)
        except Exception as e: 
            print(e)
    
def unprocess1d(modified, preprocess1d_output, range_low:float=0.1, range_high:float=1.0): 
    method_applied = preprocess1d_output[0]

    x = None 

    if method_applied == 'as-is': 
        x = modified 

    elif method_applied == 'normalize': 
        applied_max = preprocess1d_output[2]
        applied_min = preprocess1d_output[1]
        
        x = (((modified-range_low)/(range_high-range_low))*(applied_max-applied_min))+applied_min
        
    elif method_applied == 'standardize':
        mean = preprocess1d_output[1]
        sigma = preprocess1d_output[2]
        x = (modified*sigma)+mean 
    
    elif method_applied == 'median': 
        x = modified*preprocess1d_output[1]

    else: 
        raise Exception('')

    return x 

original = np.array([1,2,3,3,3,2,2,1,3,34,5,54,3,2,22,3,4,52,32,3])

modified, (method_applied, min_value, max_value) = preprocess1d(original, 'normalize')

x = unprocess1d(modified, (method_applied, min_value, max_value))

print(original/x)