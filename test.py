from qpoml import qpo 
from qpoml import observation 
from qpoml import collection

from qpoml import plotting 

q = qpo(1, 0.2, 0.1, type='A')
#print(q.properties)

#print(plotting.test())

import pandas as pd
import warnings 

df = pd.DataFrame(list(zip(['11212', '12121'], ['Maxi', 'Swift'])), columns=['observation_ID', 'object_type'])

observations = collection(df)

warnings.warn('make collections object subscriptable')
#print(observations.observations[0].observation_ID)

obs = observation(observation_ID='31321', hardness=5, rms=2, Tin=0.5)
#print(obs.features)

from qpoml import utilities 

utilities.lol([q,q], [q,q,q])
