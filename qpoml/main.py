import numpy as np
import pandas as pd
import warnings 


warnings.warn('in the spirit of minimum viable product, for now only predict qpos based on observations, or observations on qpos')
warnings.warn('TELL HRL ABOUT MVP!!')

class qpo: 

    # https://docs.astropy.org/en/stable/api/astropy.modeling.functional_models.Lorentz1D.html?highlight=lorentz1d

    def __init__(self, observation_ID:str, frequency:float, object_type:str=None, object_name:str=None, 
                 width:float=None, 
                 amplitude:float=None, Q:float=None) -> None:

        warnings.warn('need to do pre conditions')

        #self.__class__ = qpo
        self.observation_ID = observation_ID
        self.object_type = object_type 
        self.object_name = object_name 
        self.frequency = frequency 
        self.width = width 
        self.amplitude = amplitude 
        self.Q = Q

        keys = ['observation_ID', 'object_type', 'object_name', 'frequency', 'width', 'amplitude', 'Q']
        vals = [observation_ID, object_type, object_name, frequency, width, amplitude, Q] 
        self.properties = dict(zip(keys, vals))

    def plot(ax=None): 
        pass 
   
class context:
    
    def __init__(self, observation_ID:str, object_type:str=None, object_name:str=None, spectrum_channels=None, spectrum_energies=None, spectrum_intensities=None, **kwargs):
        
        self.observation_ID = observation_ID
        self.object_type = object_type
        self.object_name = object_name
        self.spectrum_channels = spectrum_channels
        self.spectrum_energies = spectrum_energies
        self.spectrum_intensities = spectrum_intensities
        self.features = kwargs 

        keys = ['observation_ID', 'object_type', 'object_name', 'spectrum_channels', 'spectrum_energies', 'spectrum_intensities']
        values = [observation_ID, object_type, object_name, spectrum_channels, spectrum_energies, spectrum_intensities]

        properties = dict(zip(keys, values))

        properties.update(kwargs)

        self.properties = properties 

        if len(kwargs)>0: 
            if spectrum_channels!=None or spectrum_energies!=None or spectrum_intensities!=None: 
                raise Exception('at the moment a context object can only be populated with scalar kwargs *or* spectrum, not both (this will be implemented in the future)') 

class collection: 
    def __init__(self, random_state:int=42) -> None:
        self.random_state = random_state
        self.loaded = False
        self.evaluated = False

    ## INIT ROUTINES ## 
    def load(self, qpo_df:pd.DataFrame=None, context_df:pd.DataFrame=None, qpo_list:list=None, context_list:list=None, qpo_preprocess:dict=None, context_preprocess:dict=None, test_size:int=0.9, shuffle:bool=True, stratify:str=None, hyper_tuning:dict=None): 
        
        from qpoml.utilities import normalize, standardize  
        from sklearn import preprocessing
        from collections import Counter

        self.qpo_df = qpo_df
        self.context_df = context_df
        self.qpo_list = qpo_list 
        self.context_list = context_list 
        self.qpo_preprocess_dict = qpo_preprocess
        self.context_preprocess_dict = context_preprocess
        self.test_size = test_size
        self.shuffle = shuffle
        self.stratify = stratify
        self.hyper_tuning = hyper_tuning

        self.qpo_categorical_key = {}  
        self.context_categorical_key = {}

        if qpo_list!=None and context_list!=None: 
            
            if len(qpo_list)!=len(context_list): 
                raise Exception('')

            else: 
                
                qpo_df = pd.DataFrame(columns=list(qpo_list[0].properties.keys()))
                context_df = pd.DataFrame(columns=list(context_list[0].properties.keys()))
                counter = 0
                for qpo, context in zip(qpo_list, context_list): 
                    qpo_df.loc[counter] = list(qpo.properties.values()) 
                    context_df.loc[counter] = list(context.properties.values())
                    counter+=1

                self.qpo_df = qpo_df 
                self.context_df = context_df                

        if len(self.qpo_df.index)!=len(self.context_df.index):
            raise Exception('')
        
        # FIX: need to fix context input and methods so it can work with spectral channels! 

        # remove columns filled with Nones from DFs         
        qpo_df = qpo_df.drop(columns=[col_name for col_name in list(qpo_df) if None in np.array(qpo_df[col_name])])  
        self.qpo_df = qpo_df

        context_df = context_df.drop(columns=[col_name for col_name in list(context_df) if None in np.array(context_df[col_name])])  
        self.context_df = context_df

        qpo_df_preprocessed = qpo_df
        context_df_preprocessed = context_df

        warnings.warn('future idea: replace iterating thru dataframes per https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas/55557758#55557758')
        warnings.warn('optimize loops with cython or numba?')


        def preprocess_df(self, input_df:pd.DataFrame, preprocess_dict:dict, obj_type:str): 
            for col_name in list(input_df): # not id columns tho!
                col_name = col_name
                if col_name != 'observation_ID': # if object type or object name are defined, then they are included as trainable params. ID is not tho! 
                    preprocess_step = preprocess_dict[col_name]
                    if type(preprocess_step)==list: 
                        # min max based on limits 
                        input_df[col_name] = normalize(input_df[col_name], preprocess_step[0], preprocess_step[1])[0]
                        # reccomended 

                    elif type(preprocess_step)==str: 
                        if preprocess_step=='standardize': 
                            input_df[col_name] = standardize(input_df[col_name])[0]
                        elif preprocess_step=='normalize': 
                            input_df[col_name] = normalize(input_df[col_name])[0]
                        elif preprocess_step=='categorical': # applies ordinal categorical feature encoding ... future: one - hot? 
                            column = np.array(input_df[col_name]) 
                            encoder = preprocessing.LabelEncoder()
                            encoder.fit(column)
                            encoded_classes = encoder.classes_ # first appearance gets 0, then comes up 
                            if obj_type == 'qpo':
                                self.qpo_categorical_key[col_name] = encoded_classes
                            elif obj_type == 'context': 
                                self.context_categorical_key[col_name] = encoded_classes 
                            
                            encoded_column = encoder.transform(column)
                            input_df[col_name] = encoded_column

                            # append qpo_categorical_keys_df to self? and context_categorical_keys_df to self? 
                            # So I only need to work with preprocessed versions until e.g. plotting when I can prettify them

                        elif preprocess_step == 'as-is': 
                            input_df[col_name] = input_df[col_name]
                        
                        else: 
                            raise Exception('')

                    else: 
                        raise Exception('')

            if obj_type=='qpo': 
                self.qpo_df_preprocessed = qpo_df_preprocessed
            
            elif obj_type=='context':
                self.context_df_preprocessed = context_df_preprocessed

        preprocess_df(self, input_df=qpo_df_preprocessed, preprocess_dict=qpo_preprocess, obj_type='qpo')
        preprocess_df(self, input_df=context_df_preprocessed, preprocess_dict=context_preprocess, obj_type='context')
        
        # need to note: at present, the package does not support missing values....????? 
        
        # fix this so it pads with zeros? what's the best way to handle missing data? 
        # min max scaler 0.1-1.1 so 0 can be no obs? 

        num_qpo_props = len(list(qpo_df_preprocessed.iloc[[0]]))-1
        ids = qpo_df_preprocessed['observation_ID'] 
        max_simultaneous_qpos = Counter(ids).most_common(1)[0][1]
        vector_length = max_simultaneous_qpos*num_qpo_props
        
        qpo_tensor_preprocessed = np.array([]).reshape(0,vector_length)

        loaded_qpo_ids = []
        for observation_ID in np.array(qpo_df_preprocessed['observation_ID']): 
            if observation_ID not in loaded_qpo_ids: 
                # sort ascending to frequency if multiple qpos 
                slice_indices = np.where(qpo_df_preprocessed['observation_ID']==observation_ID)[0]
                sliced_qpo_df = qpo_df_preprocessed.iloc[slice_indices]
                sliced_qpo_df = sliced_qpo_df.drop(columns=['observation_ID'])
                sliced_qpo_df.sort_values(by=['frequency'])

                qpo_temp_list = list(np.array([sliced_qpo_df.loc[i].tolist() for i in sliced_qpo_df.index]).flatten())
                
                pad_length = vector_length-len(qpo_temp_list)
                if pad_length>0: 
                    qpo_temp_list = qpo_temp_list+(vector_length-len(qpo_temp_list))*[0]
                qpo_tensor_preprocessed = np.vstack((qpo_tensor_preprocessed, qpo_temp_list))

                loaded_qpo_ids.append(observation_ID)


        self.qpo_tensor_preprocessed = qpo_tensor_preprocessed

        # an observation can only have one context object, but it can be matched with multiple qpo objects 

        # in qpo tensor, for now each qpo will be a vector that repeats for new qpos for an object. 
        # e.g. if an observation is associated with two simultaneous qpos, its vector in the qpo tensor could be [first_freq, first_width, first_power, second_freq, second_width, second_power]
        # later today: incorporate fundamental and harmonic stuff...for now qpos will be sorted from low to high frequency

        # incorporate dates some how? they shouldn't be included in training/testing tho imo

        # normalize columns in dataframe then apply same reshaping here as above...add qpo_df_preprocessed and context_df_preproccessed to attributes
        # it is very convienient how this will handle all the complexity under the hood ... e.g. simultaneous qpos for one pds

        self.loaded = True 

    def evaluate(self, x:str='context', y='qpo', model:str=None, optimize:str=None, new_collection=None):
        
        self.check_loaded('evaluate')

            #self.evaluated = True

    ## UTILITIES ##

    # post load #
    def correlation_matrix(self): 
        self.check_loaded('correlation_matrix') 
    def dendogram(self):
        self.check_loaded('dendogram')
    def calculate_vif(self): # from statsmodels.stats.outliers_influence import variance_inflation_factor as vif; only applied to context
        self.check_loaded('calculate_vif')
    def vif(self, cutoff_value): # remove columns from context with vif more than cutoff ... note that multicolinearity is not a concern for pure accuracy, mainly a concern when dealing with feature importances, which is important in elucidating useful takeaways; only applied to context 
        self.check_loaded('cull_vif')
    def pca_transform(self): # once these happen, context_df and arrays are changed to place holder names with transformed vectors;  only applied to context
        self.check_loaded('pca_transform')
    def mds_transform(self): # only applied to context
        self.check_loaded('mds_transform')
    def rebin_spectrum(self): # if context columns are spectral channels, rebin will rebin them and assign new column names while associating ranges to them. 
        self.check_loaded('rebin_spectrum')

    # post evaluation #
    def confusion_matrix(self): 
        self.check_evaluated('confusion_matrix')
    def results_regression(self):
        self.check_evluated('results_regression')
    def feature_importances(self):
        self.check_evaluated('feature_importances')

    ## PLOTTING ## 

    # post load # 
    def plot_correlation_matrix(self): # qpo, context, both 
        self.check_loaded('plot_correlation_matrix') 
    def plot_pairplot(self, dendogram=False): # because seaborn annoys me; qpo, context, both  
        self.check_loaded('plot_pairplot') 
    def plot_dendogram(self):# qpo, context
        self.check_loaded('plot_dendogram') 
    def plot_vif(self): 
        self.check_loaded('plot_vif')  
    def plot_pca_transform(self, c='col_name'): # no colors would make sense here related to context, only those related to qpo 
        self.check_loaded('plot_pca_transform') 
    def plot_mds_transform(self): 
        self.check_loaded('plot_mds_transform')  

    # post evaluation # 
    def plot_confusion_matrix(self): 
        self.check_evaluated('plot_confusion_matrix')
    def plot_results_regression(self, color=''): # column name from either context or parameter name from qpo. if categorical, will plot and label by categories. otherwise, will plot by continuous color (add add color bar?))  
        self.check_evaluated('plot_results_regression')
    def plot_feature_importances(self, color=''):
        # option to only plot top n important features...default plots feature importances for all 
        # if set to channels, make sure there is an option to label the bars by spectral ranges! probably like low\nhigh or just start, as end can be inferred from next start
        # future idea: allow for "discontinuities" in spectral channels (e.g. channels goes from 1.4-4.5 kev merged with 6.7-9.4 keV)
        self.check_evaluated('plot_feature_importances')
        

    ## HELPERS ##
    def check_loaded(self, function:str):
        if not self.evaluated: 
            raise Exception('collection must be loaded before '+function+'() function can be accessed') 
    
    def check_evaluated(self, function:str):
        if not self.evaluated: 
            raise Exception('collection must be evaluated before '+function+'() function can be accessed') 

class pds: 
    def __init__(self, frequency, power):

        self.frequency = frequency
        self.power = power 

    def find_qpos(method:str='default'): 
        pass 

    def plot(self, ax=None): 
        import matplotlib.pyplot as plt

        if ax==None: 

            fig, ax = plt.subplots()

        ax.scatter(self.frequency, self.power)