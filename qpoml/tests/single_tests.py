def test_load_basic():
    try: 
        from qpoml.main import collection

        spectrum_context = './research and development/example_spectrum.csv'
        scalar_context = './research and development/example_scalar.csv'
        qpo = './research and development/example_qpo.csv'
        order_qpo = './research and development/example_qpo_order.csv'

        qpo_preprocess = {'frequency':[0.01,20], 'width':[0.001,4], 'amplitude':[0.001, 5]}

        
        # eurostep qpo approach with spectrum by-row that's rebinned to 
        collection_one = collection()
        collection_one.load(qpo_csv=qpo, context_csv=spectrum_context, context_type='spectrum', 
                        context_preprocess='median', qpo_preprocess=qpo_preprocess, qpo_approach='eurostep', 
                        spectrum_approach='by-row', rebin=2)

        # single qpo approach with spectrum by-column 
        collection_two = collection()
        collection_two.load(qpo_csv=qpo, context_csv=spectrum_context, context_type='spectrum', 
                            context_preprocess='median', qpo_preprocess=qpo_preprocess, qpo_approach='single', 
                            spectrum_approach='by-column')

        # single qpo approach with order ... scalar context 
        collection_three = collection()
        collection_three.load(qpo_csv=order_qpo, context_csv=scalar_context, context_type='scalar', 
                        context_preprocess={'gamma':[1,4], 'tin':[0.1,3]}, qpo_preprocess=qpo_preprocess, qpo_approach='single')
    
        assert True

    except: 
        assert False 

def test_evaluation_single(base_dir:str='/ar1/PROJ/fjuhsd/personal/thaddaeus/qpo-ml-work/src/QPOML/'): 
    try: 
        from qpoml.main import collection
        import matplotlib.pyplot as plt 

        spectrum_csv = base_dir+'qpoml/tests/references/fake_generated_spectrum.csv'
        qpo_csv = base_dir+'qpoml/tests/references/fake_generated_qpos.csv'
        collection_one = collection()
        collection_one.load(qpo_csv=qpo_csv, context_csv=spectrum_csv, context_type='spectrum', context_preprocess='median', 
                            qpo_preprocess={'frequency':[1,16], 'width':[0.1,1.6], 'amplitude':[1,6]}, qpo_approach='single', 
                            spectrum_approach='by-row', rebin=3)

        from sklearn.ensemble import RandomForestRegressor

        regr = RandomForestRegressor()

        collection_one.evaluate(model=regr, model_name='RandomForestRegressor', evaluation_approach='default')

        #print(collection_one.performance_statistics())

        fig, axs = plt.subplots(2,3, figsize=(20,12))

        collection_one.plot_correlation_matrix(ax=axs[0,0])
        collection_one.plot_dendrogram(ax=axs[0,2])
        collection_one.plot_vif(ax=axs[1,0])
        collection_one.plot_results_regression(feature_name='frequency', which=[0,1], ax=axs[1,1])
        collection_one.plot_feature_importances(kind='tree-shap', ax=axs[1,2])

        plt.tight_layout()

        plt.savefig(base_dir+'qpoml/tests/outputs/spectrum_input_basic_plots.png', dpi=150)
        plt.clf()
        plt.close()

        fig, ax = plt.subplots(figsize=(6,6))

        collection_one.plot_pairplot(ax=ax)
        plt.savefig(base_dir+'qpoml/tests/outputs/spectrum_input_pair_plot.png', dpi=150)
        plt.clf()
        plt.close()

        scalar_collection_csv = base_dir+'qpoml/tests/references/fake_generated_scalar_context.csv'
        
        collection_two = collection()
        context_dict = {'gamma':[1.0,3.5],'T_in':[0.1,2.5],'hardness':[0,1]}
        collection_two.load(qpo_csv=qpo_csv, context_csv=scalar_collection_csv, context_type='scalar', context_preprocess=context_dict, 
                            qpo_preprocess={'frequency':[1,16], 'width':[0.1,1.6], 'amplitude':[1,6]}, qpo_approach='single')

        regr = RandomForestRegressor()

        collection_two.evaluate(model=regr, model_name='RandomForestRegressor', evaluation_approach='k-fold', folds=5, repetitions=4)

        #print(collection_one.performance_statistics())

        fig, axs = plt.subplots(2,3, figsize=(20,12))

        collection_two.plot_correlation_matrix(ax=axs[0,0])
        collection_two.plot_dendrogram(ax=axs[0,2])
        collection_two.plot_vif(ax=axs[1,0])
        collection_two.plot_results_regression(feature_name='frequency', which=[0,1], ax=axs[1,1])
        collection_two.plot_feature_importances(kind='tree-shap', ax=axs[1,2])

        plt.tight_layout()

        plt.savefig(base_dir+'qpoml/tests/outputs/scalar_input_three_fold_basic_plots_.png', dpi=150)
        plt.clf()
        plt.close()

        fig, ax = plt.subplots(figsize=(6,6))

        collection_two.plot_pairplot(ax=ax)
        plt.savefig(base_dir+'qpoml/tests/outputs/scalar_input_three_fold_pair_plot.png', dpi=150)
        plt.clf()
        plt.close()

        fig, ax = plt.subplots()

        collection_two.plot_fold_performance(statistic='mse', ax=ax)
        plt.savefig(base_dir+'qpoml/tests/outputs/fold_performance_plot_test.png', dpi=150)
        plt.clf()
        plt.close()

        assert True

    except: 
        assert False

test_evaluation_single()

def test_categorical_scalar_and_grid_search(): 
    try: 
        from qpoml import collection
        from sklearn.ensemble import RandomForestRegressor

        qpo_csv = './qpoml/tests/references/fake_generated_qpos.csv'
        scalar_collection_csv = './qpoml/tests/references/fake_generated_scalar_context_with_categorical.csv'
        
        collection_two = collection()
        context_dict = {'gamma':[1.0,3.5],'T_in':[0.1,2.5],'qpo_class':'categorical'}
        collection_two.load(qpo_csv=qpo_csv, context_csv=scalar_collection_csv, context_type='scalar', context_preprocess=context_dict, 
                            qpo_preprocess={'frequency':[1,16], 'width':[0.1,1.6], 'amplitude':[1,6]}, qpo_approach='single')

        regr = RandomForestRegressor()

        collection_two.evaluate(model=regr, model_name='RandomForestRegressor', evaluation_approach='k-fold', folds=5, repetitions=4)

        random_forest_params = {'n_estimators':[100,200], 'min_samples_split':[2,3,4]}
        best_configuration, _, _ = collection_two.grid_search(model=regr, parameters=random_forest_params)

        with open('./qpoml/tests/outputs/grid_search_best_model.txt', 'w') as f: 
            f.write(','.join(best_configuration.keys())+'\n')
            f.write(','.join(best_configuration.values())+'\n')

        assert True 
    except: 
        assert False 

#test_categorical_scalar_and_grid_search()