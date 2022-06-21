def first_version(observations=100): 
    import numpy as np
    import pandas as pd 

    qpo_ids = []
    freqs = []
    widths = []
    norms = []

    context_rows = []

    for i in range(observations): 
        context_observation_IDs = []
        context_rows = []

        id = str(i)
        if len(id)==1: 
            id = 'ID_00'+id
        else: 
            id = 'ID_0'+id

        context_observation_IDs.append(id)

        def random_intensity(): 
            return int(20*np.random.random(1)+1) #[1-20)
        
        context_row = [id]+[random_intensity() for k in range(20)] # twenty spectrum channels 
        context_rows.append(context_row)

        for j in range(np.random.randint(0, 3)): 
            qpo_ids.append(id)
            freq = 1+15*np.random.random(1)[0] # [1,16)
            freqs.append(freq)
            width = 0.1*freq # [0.1,1.6) 
            widths.append(width)
            norm = 1+5*np.random.random(1)[0] #[1.6)
            norms.append(norm)
    
    qpo_df = pd.DataFrame()
    qpo_df['observation_ID'] = qpo_ids
    qpo_df['frequency'] = freqs 
    qpo_df['width'] = widths 
    qpo_df['amplitude'] = norms 

    print(context_rows)
    print(context_observation_IDs)

    context_df = pd.DataFrame(context_rows, columns=['observation_ID']+['channel_'+str(i) for i in range(20)])

    qpo_df.to_csv('./qpoml/tests/current/references/fake_generated_qpos.csv', index=False)
    context_df.to_csv('./qpoml/tests/current/references/fake_generated_spectrum.csv', index=False)

first_version()
            
        

        
