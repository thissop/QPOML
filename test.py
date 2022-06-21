import numpy as np

import pandas as pd 

qpo_tensor = np.array([[2.562282895,0.230605,0.00205078, 0, 0, 0, 0, 0, 0], 
              [2.1,0.210978,0.002437472,2.506290149,0.240978,0.00337472, 0, 0, 0], 
              [2.648626213,0.238384,0.00271215,3.648626213,0.338384,0.00371215,3.02,0.308384,0.0041215]])

print(qpo_tensor)
print(qpo_tensor.T)

quit()

transposed = qpos.T
#print(transposed)

for i in range(max_simultaneous+1): 
    print('lol')
    combined_indices = []
    for j in range(0, num_features+1, num_features): 
        idx = i+j
        combined_indices.append(idx)

    flat = transposed[combined_indices].flatten()

    low, high = (np.min(flat), np.max(flat))

    for idx in combined_indices:
        x = transposed[idx] 
        transposed[idx] = (x-low)/(high-low)

print(transposed.T)
    
