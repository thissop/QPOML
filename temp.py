import numpy as np
from qpoml.utilities import preprocess1d as pre
from qpoml.utilities import rev

x = np.array([1,2,3,4,5])
trans, _ = pre(x, 'normalize')
print(trans)
print(rev(trans, ('normalize', 1, 5)))