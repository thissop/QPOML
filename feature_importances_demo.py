def demo(): 

    from xgboost import XGBRegressor
    from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
    from sklearn.datasets import make_regression
    from sklearn.inspection import permutation_importance
    import matplotlib.pyplot as plt 
    import shap
    import numpy as np
    from sklearn.tree import DecisionTreeRegressor
    import os 


    X, y = make_regression(n_features=4, n_informative=2,
                        random_state=0, shuffle=False)

    feature_names = [0,1,2,3]

    # feature importances options (all models): permutation, kernel-shap (default) 
    # feature importances (tree based models): gini, tree-shap (default)
    # feature importances for xgboost only: weight, gain, cover, total_gain, total_cover  

    models = []
    model_names = []

    # decision tree # 
    tree = DecisionTreeRegressor()
    tree.fit(X,y)

    #models.append(tree)
    #model_names.append('DecisionTreeRegressor')

    # random forest # 
    random_forest = RandomForestRegressor()
    random_forest.fit(X, y)

    #models.append(random_forest)
    #model_names.append('random_forest')

    # gradient boosted #
    gradient_boosted = GradientBoostingRegressor()
    gradient_boosted.fit(X,y)
    #models.append(gradient_boosted)
    #model_names.append('GradientBoostingRegressor')

    # xgboost # 
    xgb = XGBRegressor(n_estimators=100)
    xgb.fit(X, y)
    
    #models.append(xgb)
    #model_names.append('xgboost')

    # adaboost #
    ada_boost = AdaBoostRegressor()
    ada_boost.fit(X,y)
    models.append(ada_boost)
    model_names.append('AdaBoost')

    #def feature_importances(model, )

    feature_nums = np.array(list(range(0,4)))
    for model, model_name in zip(models, model_names): 
        fig, axs = plt.subplots(2,3, figsize=(8,5))

        fig.suptitle(model_name)

        sort_idx = np.argsort(model.feature_importances_)[::-1]

        ax = axs[0,0]
        ax.barh(feature_nums[sort_idx], np.array(model.feature_importances_)[sort_idx])
        ax.set(title=model_name+' Gini Importances')

        perm_importance = permutation_importance(model,X, y)['importances_mean']
        sort_idx = np.argsort(perm_importance)[::-1]
        ax = axs[0,1]
        ax.barh(feature_nums[sort_idx], np.array(perm_importance)[sort_idx])
        ax.set(title=model_name+'Permutation Importances')

        if model_name == 'xgboost': 

            ax = axs[0,2]
            scores = list(xgb.get_booster().get_score(importance_type='gain').values())
            sort_idx = np.argsort(scores)
            ax.barh(feature_nums[sort_idx], np.array(scores)[sort_idx])
            ax.set(title=model_name+'XGBOOST Gain Importances')


        ax = axs[1,0]
        ax.set(title='TreeExplainer', xlabel='|SHAP Value|')

        if shap.KernelExplainer.supports_model(model): 

            explainer = shap.TreeExplainer(model)
            shap_values = np.array(explainer.shap_values(X))
            shap_values = np.array(shap_values).T
            shap_values = np.array([np.mean(np.abs(i)) for i in shap_values])
            sorted_idx = np.argsort(shap_values)[::-1]
            ax.barh(feature_nums[sorted_idx], shap_values[sorted_idx])


        # reason for using treeshap (from the docs): Tree SHAP is a fast and exact method to estimate SHAP values for tree models and ensembles of trees, under several different possible assumptions about feature dependence.
        # i.e. treeshap is faster than kernelshap for tree based models. 
        # https://github.com/slundberg/shap/blob/master/shap/plots/_bar.py

        ax = axs[1,1]


        ax.set(title='KernelSHAP', xlabel='|SHAP Value|')
        
        if shap.KernelExplainer.supports_model(model): 

            explainer = shap.KernelExplainer(model.predict, X)
            shap_values = explainer.shap_values(X)
            shap_values = np.array(shap_values).T
            shap_values = np.array([np.mean(np.abs(i)) for i in shap_values])
            ax.barh(feature_nums, shap_values)

        # From the docs:  Kernel SHAP [is] a model agnostic method to estimate SHAP values for any model. Because it makes no assumptions about the model type, KernelExplainer is slower than the other model type specific algorithms.
        
        
        fig.tight_layout()
        plt.tight_layout()
        base_path = r"C:\Users\Research\Documents\GitHub\MAXI-J1535\code\misc\random\importance_plots"
        plt.savefig(os.path.join(base_path, model_name+'.png'), dpi=150)
        plt.clf()

demo()