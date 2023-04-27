from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=42)

# Define custom scoring function
def custom_score(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    print ("acc",acc)
    print ("acc**2",acc**2)
    formatted_f1_average = "{:.1f}".format(acc)
    print ("formatted acc",formatted_f1_average)
    return acc**2


import numpy as np
from sklearn.model_selection import KFold

def custom_grid_search(model, param_grid, X, y, cv=5, score_func=None):
    """
    Custom grid search algorithm that performs cross-validation with a custom scoring function.
    
    Parameters:
    -----------
    model : estimator object
        The model to be optimized.
        
    param_grid : dict
        Dictionary with parameters names (string) as keys and lists of parameter settings to try as values.
        
    X : array-like of shape (n_samples, n_features)
        Training input samples.
        
    y : array-like of shape (n_samples,)
        Target values.
        
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy. Possible inputs for cv are:
        - None, to use the default 5-fold cross-validation
        - integer, to specify the number of folds in a (Stratified)KFold
        - An object to be used as a cross-validation generator.
        - An iterable yielding train, test splits.
        
    score_func : callable, optional
        Custom scoring function to be used. If None, the default scoring function of the estimator is used.
        
    Returns:
    --------
    best_params : dict
        Dictionary of best parameters found during grid search.
        
    best_score : float
        Best score achieved during grid search.
    """
    
    # Create parameter grid
    param_grid = [{k: [v] if not isinstance(v, list) else v for k, v in param_grid.items()}]
    
    # Create KFold object for cross-validation
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Initialize variables for best score and parameters
    best_score = None
    best_params = {}
    
    # Loop over parameter combinations
    for params in param_grid:
        
        # Loop over cross-validation splits
        scores = []
        for train_idx, test_idx in kf.split(X):
            
            # Split data into training and test sets
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Fit model and make predictions
            model.set_params(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate score using custom scoring function
            if score_func is not None:
                score = score_func(y_test, y_pred, X_test)
            else:
                score = model.score(X_test, y_test)
                
            scores.append(score)
        
        # Calculate mean score and update best score and parameters if applicable
        mean_score = np.mean(scores)
        if best_score is None or mean_score > best_score:
            best_score = mean_score
            best_params = params
    
    return best_params, best_score


# Define parameter grid
param_grid = {
    'max_depth': [3, 5, 7, 9],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 5, 10],
}

# Create decision tree classifier object
dt = DecisionTreeClassifier()

# Create scorer object using custom scoring function
scorer = make_scorer(custom_score)

# Create GridSearchCV object
grid_search = custom_grid_search(model=dt, param_grid=param_grid, score_func=scorer , X=X, y=y, cv=5)

# Fit the GridSearchCV object to the data
grid_search.fit(X, y)

# Print the best hyperparameters and best score
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)
