def first_test(qpo_csv='./qpoml/tests/test_data/example_qpo_data.csv', 
                                        context_csv='./qpoml/tests/test_data/example_context_data.csv'):

    from qpoml import context, qpo, collection 
    import warnings
    import pandas as pd

    warnings.filterwarnings("ignore")

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
                                        hardness_ratio=row['hardness_ratio'], nthcomp_norm_before_error=row['nthcomp_norm_before_error']))

        collec = collection()
        qpo_preprocess_dict = {'frequency':[0,20],'width':'normalize','amplitude':'normalize'}
        context_preprocess_dict = {'net_source_count_rate':'normalize','hardness_ratio':'normalize', 'nthcomp_norm_before_error':'normalize'}
        collec.load(qpo_list=qpo_list, context_list = context_list, qpo_preprocess=qpo_preprocess_dict, context_preprocess=context_preprocess_dict)
        collec.evaluate(model='random-forest')

        collec.plot_feature_importances()

        assert True 

    except: 

        assert False

first_test()