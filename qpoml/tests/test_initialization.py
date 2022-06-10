def test_qpo_initialization(id:str='323232323', freq:float=1, width:float=0.4, 
                            object_type:str='BH', object_name:str='MAXI',
                            amplitude:float=1, Q:float=2):
    from qpoml import qpo 
    try: 
        q = qpo(observation_ID=id, frequency=freq, width=width, amplitude=amplitude, 
                Q=Q, object_name=object_name, object_type=object_type)
        
        assert True  

    except: 
        assert False 

def test_context_initialization(observation_ID:str='12121', object_type:str='NS', object_name:str='MAXI', 
                                disk_temperature=1.2, gamma=2.3, count_rate=1111):

    from qpoml import context

    try: 
        c = context(observation_ID='12121', object_type='NS', object_name='MAXI', 
                                disk_temperature=1.2, gamma=2.3, count_rate=1111)

        print(c.properties)
        assert True

    except: 
        assert False

def test_collection_initialization_from_lists(): 
    
    from qpoml import context, qpo, collection 
    import warnings

    warnings.filterwarnings("ignore")

    try: 

        ids = ['1', '2', '3', '4', '2']
        freqs = [1,2,3,4,4.5]
        widths = [0.1,0.15,0.2,0.3,2]
        qs = [qpo(observation_ID=i,frequency=j,width=k) for i,j,k in zip(ids,freqs,widths)]
        
        gammas = [1,2,2.3,1.5,2]
        inclinations = ['high', 'low', 'medium', 'high', 'low']
        
        cs = [context(observation_ID=i,gamma=j,inclination=k) for i,j,k in zip(ids,gammas,inclinations)]

        collec = collection()
        qpo_preprocess_dict = {'frequency':[0,10], 'width':'standardize'}
        context_preprocess_dict = {'gamma':'as-is', 'inclination':'categorical'}
        collec.load(qpo_list=qs, context_list=cs, qpo_preprocess=qpo_preprocess_dict, context_preprocess=context_preprocess_dict)
        
        print(collec.qpo_df)
        print(collec.context_df)
        print(collec.qpo_tensor_preprocessed)
        
        assert True

    except: 

        assert False 

test_collection_initialization_from_lists()