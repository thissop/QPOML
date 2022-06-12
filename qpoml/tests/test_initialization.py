def test_qpo_and_context_context_both_spectrum_and_scalars(qpo_csv='./qpoml/tests/test_data/example_qpo_data.csv', 
                                        context_csv='./qpoml/tests/test_data/example_context_data.csv'):

    from qpoml import context, qpo, collection 
    import warnings
    import pandas as pd

    warnings.filterwarnings("ignore")

    spectrums_path = './qpoml/tests/test_data/spectrum_CSVs/'

    try: 
        
        qpo_list = []
        qpo_df = pd.read_csv(qpo_csv)
        for index in qpo_df.index: 
            row = qpo_df.loc[index]
            qpo_list.append(qpo(observation_ID=row['observation_ID'], frequency=row['frequency'], width=row['width'],
                                amplitude=row['amplitude']))

        context_df = pd.read_csv(context_csv)
        context_list = []
        for index in context_df.index: 
            row = context_df.loc[index]
            context_list.append(context(observation_ID=row['observation_ID'], net_source_count_rate=row['net_source_count_rate'], 
                                        hardness_ratio=row['hardness_ratio'], nthcomp_norm_before_error=row['nthcomp_norm_before_error'], 
                                        random_class=row['object_type']))

        collec = collection()
        qpo_preprocess_dict = {'frequency':[0,20],'width':'normalize','amplitude':'normalize'}
        context_preprocess_dict = {'net_source_count_rate':'normalize','hardness_ratio':'normalize', 'nthcomp_norm_before_error':'normalize', 'random_class':'categorical'}
        collec.load(qpo_list=qpo_list, context_list = context_list, qpo_preprocess=qpo_preprocess_dict, context_preprocess=context_preprocess_dict)
        
        #print(collec.qpo_matrix[0])
        #print(collec.context_df)
        #print(collec.context_categorical_key)
        #print(collec.return_context_list()[0])
        #print(collec.return_qpo_list()[0])
        
        #print('#### SECOND TEST BEGINNING ####')

        qpo_list = []
        qpo_df = pd.read_csv(qpo_csv)
        for index in qpo_df.index: 
            row = qpo_df.loc[index]
            qpo_list.append(qpo(observation_ID=row['observation_ID'], frequency=row['frequency'], width=row['width'],
                                amplitude=row['amplitude']))

        context_list = []
        for id in pd.read_csv('./qpoml/tests/test_data/ids_list.csv')['observation_ID']: 
            spectrum = pd.read_csv(spectrums_path+id+'.csv')
            context_list.append(context(observation_ID=id, rebin_spectrum=7, spectrum=spectrum))

        context_list = context_list+context_list


        qpo_preprocess_dict = {'frequency':[0,20],'width':'normalize','amplitude':'normalize'}
        collec_two = collection()
        collec_two.load(qpo_list=qpo_list, context_list=context_list, qpo_preprocess=qpo_preprocess_dict, context_preprocess='normalize')
        #print(collec_two.qpo_matrix[0])
        #print(collec_two.spectrum_matrix[0])
        #print(collec_two.spectrum_matrix.shape)
        #print(collec_two.return_context_list()[0])
        #print(collec_two.return_qpo_list()[0])

        #print(collec_two.qpo_df)

        assert True

    except: 

        assert False 

def check_all_plots(): 
    
    plots_dir = './qpoml/tests/test_plots/'

    

    pass 


