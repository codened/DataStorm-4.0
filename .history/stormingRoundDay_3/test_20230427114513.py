import pandas as pd
from sklearn import svm
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

# Example dataset
data = {'X': [1, 2, 3, 4, 5],
        'y': [1, 0, 1, 0, 1]}
df = pd.DataFrame(data)

# Define custom scoring function
def custom_scorer(y_true, y_pred):
    # Count number of correct predictions
    num_correct = sum([1 for i in range(len(y_true)) if y_true[i] == y_pred[i]])
    # Calculate accuracy
    accuracy = num_correct / len(y_true)
    # Return accuracy as score
    return accuracy

# Set up SVM model
model = svm.SVC()

# Define hyperparameters to search over
param_grid = {'C': [0.1, 1, 10],
              'gamma': [0.1, 1, 10]}

# Define grid search with custom scoring function
grid_search = GridSearchCV(model, param_grid, scoring=make_scorer(custom_scorer) , cv=3)

# Fit grid search to data
grid_search.fit(df[['X']], df['y'])

# Print best score and hyperparameters
print("Best score:", grid_search.best_score_)
print("Best parameters:", grid_search.best_params_)
