
def inverse_encoder_transform(x, original_classes): 
    classes_dict = dict(zip(list(range(len(original_classes))), original_classes))
    x = np.array([classes_dict[i] for i in x])
    return x 
    
    # for best practice, should they only do standardization or normalization for all continuous variables? 

warnings.warn('include option to do standardize or normalize data')
warnings.warn('by default, float variables are normalized/standardized, whereas integer variables are not')


## fix below, add to collec ## 
## add proportion arg that randomly searches thru that proportion of grid only as well##
def run_grid_search(grid, X, y, n_splits: int = 10, model: str = 'randomforest', proportion:float=1): 
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    
    skf = StratifiedKFold(n_splits=n_splits)

    if model=='randomforest': 

        grid_dict = {'n_estimators':[50, 100, 500, 1000],
                'min_samples_split':[2,4,6,8],
                'min_samples_leaf':[1,2,3]}

        keys = grid_dict.keys()
        grid_matrix = []
        grid_scores = []
        for i in grid_dict[keys[0]]: 
            for j in grid_dict[keys[1]]: 
                for k in grid_dict[keys[2]]: 
                    combination = [i,j,k]
                    grid_matrix.append(combination)

        for combination in grid_matrix: 
            scores = []
            for train_index, test_index in skf.split(X, y):

                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                regr = RandomForestRegressor(n_estimators=combination[0], 
                                             min_samples_split=combination[1], 
                                             min_samples_leaf=combination[2])
                regr.fit(X_train,y_train)

                predictions = regr.predict(X_test)
                
                score = regr.score(y_test, predictions)
                scores.append(score)
            
            grid_scores.append(np.mean(scores))
    
        return grid_matrix, grid_scores

    elif model=='linear':
        scores = []
        for train_index, test_index in skf.split(X, y):

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            regr = LinearRegression()
            regr.fit(X_train,y_train)

            predictions = regr.predict(X_test)
            
            score = regr.score(y_test, predictions)
            scores.append(score)

        return scores, np.mean(scores)