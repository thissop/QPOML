import numpy as np
import pandas as pd
import warnings 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

plt.style.use('seaborn-darkgrid')
plt.rcParams['font.family']='serif'
mono_cmap = 'Blues'
bi_cmap = sns.diverging_palette(220, 10, as_cmap=True)

class qpo: 
    def __init__(self, observation_ID:str, frequency:float, width:float=None, amplitude:float=None, Q:float=None):

        warnings.warn('need to do pre conditions')

        self.observation_ID = observation_ID
        self.frequency = frequency 
        self.width = width 
        self.amplitude = amplitude 
        self.Q = Q

        keys = ['observation_ID', 'frequency', 'width', 'amplitude', 'Q']
        vals = [observation_ID, frequency, width, amplitude, Q] 
        mask = [i for i in range(len(vals)) if type(vals[i])!=type(None)]
        keys = [keys[i] for i in mask]
        vals = [vals[i] for i in mask]

        # normalize 0.1-1.1??? look into this

        self.properties = dict(zip(keys, vals))

        features = dict(self.properties)
        features.pop('observation_ID')
        self.features = features

    def plot(ax=None): 
        pass 
   
class context:
    
    def __init__(self, observation_ID:str, spectrum=None, rebin_spectrum:int=None, **kwargs):
        
        if len(kwargs)>0 and type(spectrum)!=type(None): 
            raise Exception('')
        elif len(kwargs)==0 and type(spectrum)==type(None): 
            raise Exception('')

        self.observation_ID = observation_ID
        self.properties = None
        self.features = kwargs # kwargs should be scalars!

        self.spectrum_df = None
        self.ordinate_col = None

        if not isinstance(spectrum, (type(None))): 
            from astropy.io import fits 

            if isinstance(spectrum, (pd.DataFrame)): 
                spectrum_df = spectrum 
                self.spectrum_df = spectrum

            else: 
                raise Exception('')
        
        if not isinstance(self.spectrum_df, (type(None))): 
            cols = list(self.spectrum_df)
            default_cols = ['energy', 'energy_range']
            ordinate_col = np.setdiff1d(cols, default_cols)[0]
        
            self.ordinate_col = ordinate_col
            
        # only need energy, energy range, and ordinate? 
        if type(rebin_spectrum)!=type(None): 

            warnings.warn('rebin is not working...need to come back to it sometime')

            orig_length = len(self.spectrum_df.index)

            rebin_energies = []
            rebin_energy_ranges = []
            rebin_ordinates = []
            
            step = int(orig_length/rebin_spectrum) 
            lower = range(0, orig_length-step, step)
            upper = np.array(lower)+step

            # warn users that last channels will be cut off 

            for i, j in zip(lower, upper): 
                rebin_energies.append(np.mean(self.spectrum_df['energy'][i:j]))
                
                flat = np.array(self.spectrum_df['energy_range'])[i:j]
                
                if type(flat[0])==str: 
                    flat = np.array([eval(i) for i in flat])
                
                flat = flat.flatten()

                rebin_energy_ranges.append((np.min(flat), np.max(flat)))

                rebin_ordinates.append(np.sum(self.spectrum_df[ordinate_col][i:j]))

            rebin_df = pd.DataFrame()
            rebin_df['energy'] = rebin_energies
            rebin_df['energy_range'] = rebin_energy_ranges
            rebin_df[ordinate_col] = rebin_ordinates
            self.spectrum_df = rebin_df

        keys = ['observation_ID', 'spectrum_df', 'ordinate_col']
        vals = [observation_ID, self.spectrum_df, self.ordinate_col]
        mask = [i for i in range(len(vals)) if type(vals[i])!=type(None)] # I was having weird numpy problems here so just list comp for now :(
        keys = [keys[i] for i in mask]
        vals = [vals[i] for i in mask]

        properties = dict(zip(keys, vals))

        properties.update(kwargs)

        self.properties = properties 

class collection: 

    # Initialize #
    def __init__(self, random_state:int=42) -> None:
        
        ### global ###

        self.random_state = random_state
        self.available_models = ['ada-boost', 'decision-tree', 'linear-ordinary', 'linear-ridge', 
                                 'neural-network', 'random-forest', 'support-vector-machine',
                                 'xgboost'] # <=== give this a getter method! 
        
        ### load initialized ### 

        # comment on ada boost to family ... aida from shield lol

        self.loaded = False
        self.evaluated = False

        self.qpo_categorical_key = {}
        self.qpo_df = None 
        self.qpo_matrix = None # will be preprocessed 
        
        self.context_df = None
        self.context_categorical_key = {}
        self.context_is_spectrum = False
        self.spectrum_matrix = None # will be preprocessed
        self.spectrum_matrix_df = None
        self.ordinate_col= None
        self.spectrum_df_template = None

        ### evaluate initialized ### 
        
        self.x_name = None  
        self.y_name = None 

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None 

        self.fit_model = None 
        self.fit_model_name = None # <== set to 'external' so plotting/misc functions that need model info won't run if this was true
        self.predictions = None 

    # Load # 
    def load(self, qpo_list:list=None, context_list:list=None, qpo_preprocess:dict=None,
             context_preprocess=None, test_size=0.9,
             shuffle:bool=True): 

        self.dont_do_twice('load')

        from sklearn.model_selection import train_test_split

        warnings.warn('future idea: right now only spectrum *or* scalars can be loaded into context...in future do both?')  

        # if context_preprocess is string, then that string will be used to a

        warnings.warn('future idea: what would be the difference between normalizing features by object or globally?')

        from qpoml.utilities import normalize, standardize  
        from sklearn import preprocessing  
        from collections import Counter  
        from qpoml import qpo, context

        # for now, end goal depending on inputs is to get qpo_list and context_list populated. these will be the things returned to the user
        # it will be really easy if I just train test split a list of indices and then apply those when returning train test split qpo_list and context_list 
        # alert users to the fact that the context_list and qpo_list attributes of the collection after loading may not be the same as the arguments passed to load with the same name,  
        # e.g. if those lists were not filled with qpo/context objects, then 
        # talk about how load includes lots of convenience optimizations so people wouldn't have to e.g. make a list of context and qpo objects individually, and then pass those to the function. 

        warnings.warn('need to fix up all my none comparisons...right now I do it three separate ways :(')

        if qpo_list!=None and context_list!=None: 
            if len(qpo_list)!=len(context_list):
                raise Exception('')

            self.context_list = context_list
            self.qpo_list = qpo_list

            ids = [context_item.observation_ID for context_item in context_list]

            indices = list(range(0,len(set(ids))))

            train_indices, test_indices = train_test_split(indices, test_size=test_size, shuffle=shuffle, random_state=self.random_state)
            
            self.train_indices = train_indices
            self.test_indices = test_indices

            context_is_spectrum = False
            if 'spectrum_df' in context_list[0].properties.keys(): 
                ordinate_col = context_list[0].ordinate_col
                self.ordinate_col = ordinate_col
                self.spectrum_df_template = context_list[0].spectrum_df.drop(columns=[ordinate_col])
                context_is_spectrum = True

            self.context_is_spectrum = context_is_spectrum

            qpo_df = pd.DataFrame(columns=list(qpo_list[0].properties.keys()))
            if not context_is_spectrum: 
                context_df = pd.DataFrame(columns=list(context_list[0].properties.keys()))
            for counter in range(len(qpo_list)): 
                qpo_df.loc[counter] = list(qpo_list[counter].properties.values()) 
                if not context_is_spectrum: 
                    context_df.loc[counter] = list(context_list[counter].properties.values())

            

            def preprocess(self, preprocess, input_df:pd.DataFrame=None, obj_type:str=None, context_list:list=None, context_is_spectrum:bool=False): 
                
                if not isinstance(input_df, (type(None))): 
                
                    for col_name in list(input_df):
                        if col_name != 'observation_ID': # if object type or object name are defined, then they are included as trainable params. ID is not tho! 
                            preprocess_step = preprocess[col_name]
                            if isinstance(preprocess_step, (list)): 
                                # min max based on limits 
                                input_df[col_name] = normalize(input_df[col_name], preprocess_step[0], preprocess_step[1])[0]
                                # recommended 

                            elif isinstance(preprocess_step, (str)): 
                                if preprocess_step=='standardize': 
                                    input_df[col_name] = standardize(input_df[col_name])[0]
                                elif preprocess_step=='normalize': 
                                    input_df[col_name] = normalize(input_df[col_name])[0]
                                elif preprocess_step=='categorical': # applies ordinal categorical feature encoding ... future: one - hot? 
                                    column = np.array(input_df[col_name]).reshape(-1,1)
                                    encoder = preprocessing.OneHotEncoder()
                                    encoder.fit(column)
                                    original_classes = encoder.categories_[0]
                                    encoded_classes = encoder.transform(original_classes.reshape(-1,1)).toarray()
                                    encoding_key = {'original_classes':original_classes, 'encoded_classes':encoded_classes}
                                    if obj_type == 'qpo':
                                        self.qpo_categorical_key[col_name] = encoding_key
                                    elif obj_type == 'context': 
                                        self.context_categorical_key[col_name] = encoding_key
                                    
                                    encoded_column = encoder.transform(column).toarray()
                                    
                                    input_df[col_name] = [arr for arr in encoded_column]

                                    # append qpo_categorical_keys_df to self? and context_categorical_keys_df to self? 
                                    # So I only need to work with preprocessed versions until e.g. plotting when I can prettify them

                                elif preprocess_step == 'as-is': 
                                    input_df[col_name] = input_df[col_name]
                                
                                else: 
                                    raise Exception('')

                            else: 
                                raise Exception('')

                elif context_list!=None and context_is_spectrum: 
                    spectrum_matrix = np.array([]).reshape(0,len(context_list[0].spectrum_df.index))

                    for context in context_list: 
                        spectrum_df = context.spectrum_df
                        ordinate_col = context.ordinate_col
                        spectral_ordinate = np.array(spectrum_df[ordinate_col])
                        spectrum_matrix = np.vstack((spectrum_matrix, spectral_ordinate))

                    if isinstance(preprocess, (list)): 
                        # min max based on limits 
                        spectrum_matrix = np.array([normalize(row, preprocess_step[0], preprocess_step[1])[0] for row in spectrum_matrix]) 
                    
                    elif isinstance(preprocess, (str)): 
                        if preprocess=='normalize': 
                            spectrum_matrix = np.array([normalize(row, np.min(spectrum_matrix), np.max(spectrum_matrix))[0] for row in spectrum_matrix])
                        elif preprocess=='standardize': 
                            spectrum_matrix = np.array([standardize(row)[0] for row in spectrum_matrix])
                        else:
                            raise Exception('')

                    else: 
                        raise Exception('')

                    self.spectrum_matrix = spectrum_matrix 

            preprocess(self, input_df=qpo_df, preprocess=qpo_preprocess, obj_type='qpo')
            self.qpo_df = qpo_df

            if context_is_spectrum: 
                preprocess(self, preprocess=context_preprocess, context_list=context_list, context_is_spectrum=True)
            else: 
                preprocess(self, input_df=context_df, preprocess=context_preprocess, obj_type='context')
                self.context_df = context_df
                
            

            # Create QPO matrix # 
            num_qpo_props = len(list(qpo_df.iloc[[0]]))-1
            ids = qpo_df['observation_ID'] 
            max_simultaneous_qpos = Counter(ids).most_common(1)[0][1]
            vector_length = max_simultaneous_qpos*num_qpo_props
            
            qpo_matrix = np.array([]).reshape(0,vector_length)

            loaded_qpo_ids = []
            for observation_ID in np.array(qpo_df['observation_ID']): 
                if observation_ID not in loaded_qpo_ids: 
                    # sort ascending to frequency if multiple qpos 
                    slice_indices = np.where(qpo_df['observation_ID']==observation_ID)[0]
                    sliced_qpo_df = qpo_df.iloc[slice_indices]
                    sliced_qpo_df = sliced_qpo_df.drop(columns=['observation_ID'])
                    sliced_qpo_df.sort_values(by=['frequency'])

                    qpo_temp_list = list(np.array([sliced_qpo_df.loc[i].tolist() for i in sliced_qpo_df.index]).flatten())
                    
                    pad_length = vector_length-len(qpo_temp_list)
                    if pad_length>0: 
                        qpo_temp_list = qpo_temp_list+(vector_length-len(qpo_temp_list))*[0]
                    qpo_matrix = np.vstack((qpo_matrix, qpo_temp_list))

                    loaded_qpo_ids.append(observation_ID)


            self.qpo_matrix = qpo_matrix

        else: 
            raise Exception()

        self.loaded = True 

    # Evaluate # 
    def evaluate(self, model, x:str='context', y='qpo'):
    
        self.check_loaded('evaluate')
        self.dont_do_twice('evaluate')

        self.x_name = x  
        self.y_name = y

        current_models = ['random-forest', 'xgboost', '']

        train_indices = self.train_indices
        test_indices = self.test_indices 

        # Train Test Split # 

        spectrum_matrix = self.spectrum_matrix 
        context_df = self.context_df    
        qpo_matrix = self.qpo_matrix 

        X_train, X_test, y_train, y_test = 4*[None]

        if x=='context': 
            if type(context_df)!=type(None):
                temp = context_df.drop(columns=['observation_ID'])
                X_train = temp.iloc[train_indices]
                X_test = temp.iloc[test_indices] 
            elif type(spectrum_matrix)!=type(None): 
                pass
            else: 
                raise Exception('')

        elif y=='context': 
            if type(context_df)!=type(None):
                temp = context_df.drop(columns=['observation_ID'])
                y_train = temp.iloc[train_indices]
                y_test = temp.iloc[test_indices]
            elif type(spectrum_matrix)!=type(None): 
                y_train = spectrum_matrix[train_indices]
                y_test = spectrum_matrix[test_indices]
            else: 
                raise Exception('')

        if x=='qpo': 
            X_train = qpo_matrix[train_indices]
            X_test = qpo_matrix[test_indices]

        elif y=='qpo':
            y_train = qpo_matrix[train_indices]
            y_test = qpo_matrix[test_indices]

        if type(None) in [type(i) for i in [X_train, X_test, y_train, y_test]]: 
            raise Exception('')

        else: 
            self.X_train = X_train 
            self.X_test = X_test 
            self.y_train = y_train 
            self.y_test = y_test 

        # Train Model # --> apply these kind of code headers throughout code when preparing to release ... use proper pattern aligned with markdown too :)

        rs = self.random_state
        fit_model_name = None
        if type(model)==str: 
            internal_model = None

            if model == 'ada-boost': 
                from sklearn.ensemble import AdaBoostRegressor 
                internal_model = AdaBoostRegressor(random_state=rs, n_estimators=100)
                fit_model_name = 'ada-boost'

            elif model == 'decision-tree': 
                from sklearn.tree import DecisionTreeRegressor
                internal_model = DecisionTreeRegressor(random_state=rs)
                fit_model_name = 'decision-tree'

            elif model == 'linear-ordinary':
                from sklearn.linear_model import LinearRegression
                internal_model = LinearRegression()
                fit_model_name = 'linear-ordinary'

            elif model == 'linear-ridge': 
                from sklearn.linear_model import Ridge
                internal_model = Ridge(alpha=0.5, random_state=rs)
                fit_model_name = 'linear-ridge'

            elif model == 'neural-network': 
                fit_model_name = 'neural-network'

            elif model == 'random-forest': 
                from sklearn.ensemble import RandomForestRegressor
                internal_model = RandomForestRegressor(random_state=rs)
                fit_model_name = 'random-forest'
            
            elif model == 'support-vector-machine':
                from sklearn.svm import LinearSVR
                internal_model = LinearSVR(random_state=rs, tol=1e-5)
                fit_model_name = 'support-vector-machine'

                warnings.warn('multiple times of svr?') # https://scikit-learn.org/stable/modules/classes.html?highlight=svm#module-sklearn.svm
            
            elif model == 'xgboost':
                import xgboost
                internal_model = xgboost.XGBRegressor()
                fit_model_name = 'xgboost'
                warnings.warn('random state?')

            else: 
                raise Exception('')

            self.fit_model_name = fit_model_name

            if type(fit_model_name)!=type(None): # FIX THIS OR DIE
                
                if fit_model_name == 'neural-network': 
                    import keras
                    from keras import models, layers 
                    import tensorflow as tf

                    warnings.warn('replace keras with tensorflow? keras is getting replaced')
                    # that ^^ could be my intro to tensorflow for nasa before nasa so I can say I've used it before (as I did in my application)

                    def get_nn_model(output_shape):
                        nn_mode = models.Sequential()
                        nn_mode.add(layers.Dense(10, activation='relu')) # fix this line
                        nn_mode.add(layers.Dense(10, activation='relu')) 
                        nn_mode.add(layers.Dense(output_shape, activation='linear'))
                        opt = tf.optimizers.Adam(learning_rate=0.001)
                        nn_mode.compile(optimizer=opt, loss='mse', metrics='mae')

                    # GET OUTPUT SHAPE!!!
                    internal_model = get_nn_model()
                    epochs = 3000
                    history = internal_model.fit(X_train, y_train, validation_split = 0.1, epochs=epochs, batch_size=25, verbose=0)
                    hist = history.history

                    warnings.warn('random state?')

                    predictions = internal_model.predict(X_test)

                    self.predictions = predictions
                    self.fit_model = internal_model 

                    train_loss = hist['loss']
                    val_loss = hist['val_loss']

                    train_mae = hist['mae']
                    val_mae = hist['val_mae']

                    warnings.warn('set global mae or mse to optimize?')

                else:
                    internal_model.fit(X=X_train, y=y_train)
                    
                    self.fit_model = internal_model 
                    predictions = internal_model.predict(X=X_test)
                    self.predictions = predictions 

        else: 
            try: 
                model.fit(X=X_train, y=y_train)
                self.fit_model = model 
                predictions = model.predict(X=X_test)
                self.predictions = predictions 
                self.fit_model_name = 'external'

            except Exception as e: 
                print(e)

            # has option to pass in initialized external model for training/predictions, but note that the plotting/utility methods are only guaranteed to work with in-built models
            # thus, X_train, X_test, y_train, y_test, trained_model, and predictions will be available to users via standard accessor methods. <=== do that! 

        self.evaluated = True













    # Utilities # <== needs better name  

    ## Accessor Methods ## 
    
    def return_qpo_list(self): 
        self.check_loaded('return_qpo_list')
        return self.qpo_list 

    def return_context_list(self): 
        self.check_loaded('return_context_list')
        return self.context_list 
    
    def return_train_test(self): 
        self.check_loaded_evaluated('return_train_test') 
        return [self.X_train, self.X_test, self.y_train, self.y_test] 
        
    def return_predictions(self): 
        self.check_loaded_evaluated('return_predictions')
        return self.predictions

    def return_trained_model(self): 
        self.check_loaded_evaluated('return_predictions') 
        return self.trained_model

    ## Miscellaneous Methods ## 
    
    ### Post Load ### 

    def correlation_matrix(self, what:str): # <== I don't think this works for spectrums? 
        self.check_loaded('correlation_matrix')

        cols = []

        context_is_spectrum = self.context_is_spectrum

        if what=='qpo-and-context': 

            if context_is_spectrum: 

                warnings.warn('need to test this lol')
                warnings.warn('I need to fix this to merge on observation_ID somehow')
                spectrum_cols = self.spectrum_df_template['energy']
                spectrum_df = pd.DataFrame(self.spectrum_matrix, columns=cols)
                
                temp_df = self.qpo_df.select_dtypes(['number'])

                for col in spectrum_cols: 
                    temp_df[col] = spectrum_df[col]
                
                cols = list(temp_df)
                corr = temp_df.corr()

            else: 
                temp_df = self.qpo_df.merge(self.context_df, on='observation_ID').select_dtypes(['number']) 
                corr = temp_df.corr() # fix for spectrum! 
                cols = list(temp_df.drop(columns=['observation_ID']))

        elif what=='context': 
            
            if context_is_spectrum: 
                cols = self.spectrum_df_template['energy']
                temp_df = pd.DataFrame(self.spectrum_matrix, columns=cols)
                corr = temp_df.corr()

            else: 
                temp_df = self.context_df.select_dtypes(['number'])
                corr = temp_df.corr()
                cols = list(temp_df)

        elif what=='qpo': 
            temp_df = self.qpo_df.select_dtypes(['number'])
            corr = temp_df.corr()
            cols = list(temp_df)

        else: 
            raise Exception('')
        
        return corr, cols

    def dendrogram(self, what:str): # rename? 
        self.check_loaded('dendrogram')

        from scipy.stats import spearmanr
        from scipy.cluster import hierarchy
        from scipy.spatial.distance import squareform

        if what=='qpo': 
            sub_df = self.qpo_df_preprocessed.drop(columns=['observation_ID']).select_dtypes(['number'])
        elif what=='context': 
            sub_df = self.context_df.drop(columns=['observation_ID']).select_dtypes(['number'])
        else: 
            sub_df = self.qpo_df_preprocessed.merge(self.context_df, on='observation_ID').select_dtypes(['number']) 
        
        cols = list(sub_df)

        # below from sklearn documentation 

        # Ensure the correlation matrix is symmetric
        corr = spearmanr(sub_df).correlation
        corr = (corr + corr.T) / 2
        np.fill_diagonal(corr, 1)

        # We convert the correlation matrix to a distance matrix before performing
        # hierarchical clustering using Ward's linkage.
        distance_matrix = 1 - np.abs(corr)
        dist_linkage = hierarchy.ward(squareform(distance_matrix))
        
        return corr, dist_linkage, cols 

    def calculate_vif(self): # <== only applied to context_df and spectrum_matrix ... complete
        self.check_loaded('calculate_vif')

        from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
        
        vif_df = pd.DataFrame()

        if type(self.context_df) != type(None): 
            temp = self.context_df.drop(columns='observation_ID')
            vif_df['VIF'] = [vif(temp.values, i) for i in range(temp.shape[1])]
            vif_df['Column'] = list(temp)

        elif type(self.spectrum_matrix) != type(None): 
            temp = self.spectrum_matrix
            temp = np.transpose(temp)
            vif_df['VIF'] = [vif(temp.values, i) for i in range(temp.shape[1])]
            vif_df['Column'] = self.spectrum_df_template['energy']
        
        else: 
            raise Exception('')

        vif_df.sort_values('VIF', ascending=True)

        return vif_df 

    ''' to do '''

    def remove_by_vif(self, cutoff_value): # remove columns from context with vif more than cutoff ... note that multicollinearity is not a concern for pure accuracy, mainly a concern when dealing with feature importances, which is important in elucidating useful takeaways; only applied to context 
        self.check_loaded('remove_by_vif')

        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

        vif_info = pd.DataFrame()
        vif_info['VIF'] = [vif(X.values, i) for i in range(X.shape[1])]
        vif_info['Column'] = X.columns
        vif_info.sort_values('VIF', ascending=True)

        mask = np.where(vif_info['VIF']<5)[0]

        ols_cols = vif_info['Column'][mask]

    def remove_from_dendrogram(self, cutoff_value): # rename? 
        pass 

    def pca_transform(self): # once these happen, context_df and arrays are changed to place holder names with transformed vectors;  only applied to context
        self.check_loaded('pca_transform') # https://github.com/thissop/MAXI-J1535/blob/main/code/machine-learning/December-%202021-2022/very_initial_sanity_check.ipynb
    
    def mds_transform(self): # only applied to context
        self.check_loaded('mds_transform')

    ### Post Evaluation ### 
    
    def results_regression(self, what:str='frequency', which:str='all'):
        self.check_evaluated('results_regression')

        from scipy.stats import linregress 

        predictions = self.predictions 
        y_test = self.y_test 

        orders = ['first', 'second', 'third', 'fourth', 'fifth',
                  'sixth', 'seventh', 'eighth', 'ninth', 'tenth']

        if self.y_name == 'qpo': 
            
            qpo_cols = list(self.qpo_df.drop(columns=['observation_ID']))
            base_index = qpo_cols.index(what)
            shift = len(qpo_cols)
            num_qpos = len(y_test[0])
            
            regression_x = np.transpose(y_test)
            regression_y = np.transpose(predictions)

            # collect what 
            idx = np.arange(base_index, len(regression_x), step=shift, dtype=int)

            regression_x, regression_y = (i[idx] for i in (regression_x, regression_y)) 

            if which == 'all': 
                regression_x = regression_x.flatten()
                regression_y = regression_y.flatten()

            else: 
                idx = orders.index(which.lower())
                regression_x, regression_y = (i[idx] for i in (regression_x, regression_y))

            m, b, r, _, _ = linregress(regression_x, regression_y) # fix, and does y test go on bottom or top? 

            return regression_x, regression_y, (m, b, r)

        elif self.y_name == 'context': 
            pass

        else: 
            raise Exception('')
        
    def feature_importances(self, kind:str='default'):

        import shap

        self.check_evaluated('feature_importances')
        
        X, y = (self.X_test, self.y_test)
        model = self.fit_model 
        model_name = self.fit_model_name 
        y_cols = np.array([]) # FIX!
        permutation_df = None

        warnings.warn('fix above!')

        importances_df = None
        shap_values = None

        if kind=='default': # fix this for shap values, etc.  ---> may not be MDI, but is default feature importance 
                if hasattr(fit_model, 'feature_importances_'): 
                    feature_importances = self.fit_model.feature_importances_
                    
        elif kind=='permutation': 
            
            from sklearn.inspection import permutation_importance

            permutation_importances = permutation_importance(estimator=model, X=X, y=y)
            feature_importances = permutation_importances['importances_mean']

            sort_idx = np.argsort(feature_importances)[::-1]

            importances_df = pd.DataFrame(permutation_importances.importances[sort_idx].T, columns=y_cols[sort_idx])
        
            warnings.warn('return diststributions themselves as third param for permutation (for box and whisker)')
        
        elif kind=='kernel-shap': 
            warnings.warn('need to implement check to see if model is supported')
            explainer = shap.KernelExplainer(model.predict, X)
            shap_values = explainer.shap_values(X)
            shap_values = np.array(shap_values).T
            feature_importances = np.array([np.mean(np.abs(i)) for i in shap_values])

        elif kind=='tree-shap': 
            warnings.warn('need to check if possible!')
            explainer = shap.TreeExplainer(model)
            shap_values = np.array(explainer.shap_values(X))
            shap_values = np.array(shap_values).T
            feature_importances = np.array([np.mean(np.abs(i)) for i in shap_values])


        else: 
            warnings.warn('don\'t need to do xgboost default in spirit of MVP')
            raise Exception('')

        if type(sort_idx)==type(None): 
            sort_idx = np.argsort(feature_importances)[::-1]
    
        if kind=='tree-shap' or kind=='kernel-shap': 
            importances_df = pd.DataFrame(shap_values, columns=y_cols[sort_idx])

        return feature_importances[sort_idx], y_cols[sort_idx], importances_df
        
    ''' to do '''

    def confusion_matrix(self): 
        self.check_evaluated('confusion_matrix')

        from sklearn.metrics import confusion_matrix, accuracy_score

        y_test_flat = self.y_test.flatten()
        predicted_classes_flat = self.predicted_classes.flatten()  

        cm = confusion_matrix(y_test_flat, predicted_classes_flat)
        acc = accuracy_score(y_test_flat, predicted_classes_flat)

        return cm, acc

    ## Plotting Methods ## 

    ### Post Load ### 
    
    def plot_correlation_matrix(self, what:str, ax=None, matrix_style:str='default', cmap=bi_cmap): 
        self.check_loaded('correlation_matrix') 
        
        # if style==steps, then only show bottom half of correlation matrix
        
        corr, cols = self.correlation_matrix(what=what)

        if ax==None: 
            fig, ax = plt.subplots()

        if matrix_style=='default': 
            sns.heatmap(corr, cmap=cmap,
                ax=ax, annot=True, annot_kws={'fontsize':'small'}, yticklabels=cols,
                xticklabels=cols, cbar_kws={"shrink": .75})

        elif matrix_style=='steps': 
            mask = np.triu(np.ones_like(corr, dtype=bool))   
            print(corr)
            print(corr[mask])
            #print(np.shape(corr))
            sns.heatmap(corr, mask=mask, cmap=cmap,
                ax=ax, annot=True, annot_kws={'fontsize':'small'}, yticklabels=cols,
                xticklabels=cols, cbar_kws={"shrink": .75})

        plt.tight_layout()
        
        plt.show()

    def plot_pairplot(self, what:str, ax=None, steps:bool=False): # because seaborn annoys me; qpo, context, both  
        self.check_loaded('plot_pairplot') 

        import matplotlib.pyplot as plt
        import seaborn as sns

        temp_df = None 
        if what=='context': 
            if self.context_is_spectrum: 
                cols = self.spectrum_df_template['energy']
                temp_df = pd.DataFrame(self.spectrum_matrix, columns=cols)
                
            else: 
                temp_df = self.context_df.drop(columns=['observation_ID'])


        elif what=='qpo': 
            temp_df = self.qpo_df.drop(columns=['observation_ID'])
        
        elif what=='qpo-and-context': 
            if self.context_is_spectrum: 
                warnings.warn('fix this')
            
            else: 
                temp_df = self.qpo_df.merge(self.context_df, on='observation_ID').select_dtypes(['number'])

        if type(temp_df)!=type(None): 

            ax = sns.pairplot(data=temp_df, corner=steps)
            plt.tight_layout()
            plt.show()

        else: 
            raise Exception('')

    def plot_dendrogram(self, what:str, ax=None):# qpo, context
        self.check_loaded('plot_dendrogram') 
        from scipy.stats import spearmanr
        from scipy.cluster import hierarchy
        from scipy.spatial.distance import squareform
        import matplotlib.pyplot as plt


        X = None 
        cols = None 
        if what=='qpo':
            X = self.qpo_df.select_dtypes(['number'])
            cols = list(self.qpo_df.drop(columns=['observation_ID'])) 
        elif what=='context': 

            X = self.context_df.select_dtypes(['number'])
            cols = list(self.context_df.drop(columns=['observation_ID']))

        external_plot = True 
        if type(ax)==type(None): 
            fig, ax = plt.subplots()
        else: 
            external_plot = False

        corr = spearmanr(X).correlation

        # Ensure the correlation matrix is symmetric
        corr = (corr + corr.T) / 2
        np.fill_diagonal(corr, 1)

        # We convert the correlation matrix to a distance matrix before performing
        # hierarchical clustering using Ward's linkage.
        distance_matrix = 1 - np.abs(corr)
        dist_linkage = hierarchy.ward(squareform(distance_matrix))
        hierarchy.dendrogram(
            dist_linkage, labels=cols, ax=ax, leaf_rotation=90
        )

        if not external_plot: 
            fig.tight_layout()

        plt.show()
    
    def plot_vif(self): 
        self.check_loaded('plot_vif')  
    
    ''' to do '''
    def plot_pca_transform(self, c='col_name'): # no colors would make sense here related to context, only those related to qpo 
        self.check_loaded('plot_pca_transform') 
    
    def plot_mds_transform(self): 
        self.check_loaded('plot_mds_transform')  

    ### Post Evaluation ### 
    
    def plot_results_regression(self, ax=None, what:str='frequency', which:str='all', xlim:list=[0,1]): # column name from either context or parameter name from qpo. if categorical, will plot and label by categories. otherwise, will plot by continuous color (add add color bar?))  
        
        # for x lim, it will also used on ylim so that the graph is 1:1...default is 0:1 for normalization 
        # declare style seperately...i.e. don't declare it if ax!=None? 
        # discuss seaborn style plotting routines?
        # but not fully seaborn because seaborn annoys me? 

        self.check_evaluated('plot_results_regression')


        import matplotlib.pyplot as plt 

        regression_x, regression_y, (m, b, r) = self.results_regression(what=what, which=which)

        if type(ax)==type(None): 
            fig, ax = plt.subplots()

        ax.scatter(regression_x, regression_y)
        
        if type(xlim)==str:
            if xlim=='min-max':
                combined = np.concatenate((regression_x, regression_y))
                xlim = [np.min(combined), np.max(combined)]
                diff = (xlim[1]-xlim[0])/20
                bounds = [xlim[0]-diff, xlim[1]+diff]
                ax.set(xlim=bounds, ylim=bounds)
        
        else: 
            ax.set(xlim=xlim, ylim=ylim)

        ax.plot(xlim, xlim, label='1:1')
        r_sq = str(round(r**2, 3))
        ax.plot(np.array(xlim), m*np.array(xlim)+b, label='Best Fit ('r'$r^2=$'+' '+r_sq+')') # math in equations! set that globally! 

        ax.set(xlabel='True '+what.capitalize(), ylabel='Predicted '+what.capitalize())

        ax.legend()

        plt.show()

    def plot_feature_importances(self, ax=None, kind:str='kernel-shap'):
        # option to only plot top n important features...default plots feature importances for all 
        # if set to channels, make sure there is an option to label the bars by spectral ranges! probably like low\nhigh or just start, as end can be inferred from next start
        # future idea: allow for "discontinuities" in spectral channels (e.g. channels goes from 1.4-4.5 kev merged with 6.7-9.4 keV)
        self.check_loaded_evaluated('plot_feature_importances')

        import matplotlib.pyplot as plt 

        # fix so it works for more context spectrum as well 

        if type(ax)==type(None): 
            fig, ax = plt.subplots()

        feature_importances, y_cols, importances_df = self.feature_importances(kind=kind)
        
        if type(importances_df) != None: 
        
            y_pos = np.arange(len(y_cols))

            ax.barh(y_pos, feature_importances, align='center')
            ax.set_yticks(y_pos, labels=y_cols)        
            ax.figure.tight_layout()

        else: 
            all_data = [feature_importances[i] for i in list(feature_importances)]
            ax.boxplot(all_data, vert=False, labels=y_cols)
        
        plt.tight_layout()
        plt.show()

    ''' to do '''    
    def plot_confusion_matrix(self, ax=None): 
        self.check_evaluated('plot_confusion_matrix')

        if ax==None: 
            fig, ax = plt.subplots(figsize=(10, 8))

        cm, acc = confusion_matrix(self)
         
        warnings.warn('won\'t work right now because those classes aren\'t defined. need to check them.')

        sns.heatmap(cm, annot=True, cmap=cmap, linewidths=.5)
        ax.set_(xlabel='Actual', ylabel='Predicted', title='Confusion Matrix\nAccuracy: '+str(round(acc, 3)))
        

        plt.show()

    ## Gotcha Methods ##

    def check_loaded(self, function:str):
        if not self.loaded: 
            raise Exception('collection must be loaded before '+function+'() function can be accessed') 
    
    def check_evaluated(self, function:str):
        if not self.evaluated: 
            raise Exception('collection must be evaluated before '+function+'() function can be accessed') 

    def check_loaded_evaluated(self, function:str): 
        self.check_loaded(function)
        self.check_evaluated(function)

    def dont_do_twice(self, function:str): 
        if function=='load' and self.loaded: 
            raise Exception('collection has already been '+function+'ed and this step cannot be repeated on the object now')
        elif function=='evaluate' and self.evaluated: 
            raise Exception('collection has already been '+function+'ed and this step cannot be repeated on the object now')    
      
class pds: 
    def __init__(self, frequency, power, qpo_list):

        self.frequency = frequency
        self.power = power 

    def find_qpos(method:str='default'): 
        pass 

    def plot(self, ax=None): 
        import matplotlib.pyplot as plt

        if ax==None: 

            fig, ax = plt.subplots()

        ax.scatter(self.frequency, self.power)

        # need to add a lot of pre-defined args so users can customize the plots more....

        if qpo_list!=None: 
            for qpo in qpo_list: 
                qpo.plot(ax=ax)