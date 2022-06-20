def make_fake_spectrums(): 
    import pandas as pd
    import numpy as np

    base = np.linspace(0,33,29)
    ranges = [(base[i], base[i+1]) for i in range(len(base)-1)]
    energies = [np.mean(arr) for arr in ranges]

    for id in pd.read_csv('./qpoml/tests/test_data/ids_list.csv')['observation_ID']: 
        df = pd.DataFrame()
        df['energy'] = energies 
        df['energy_range'] = ranges 
        df['net_count_rate'] = np.random.normal(100,10,28)

        df.to_csv('./qpoml/tests/test_data/spectrum_CSVs/'+id+'.csv', index=False)

make_fake_spectrums()