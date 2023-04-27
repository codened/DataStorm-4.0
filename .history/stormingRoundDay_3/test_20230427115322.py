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
    print ("formatted acc**2",acc**2)
    return acc**2

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
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, scoring=scorer)

# Fit the GridSearchCV object to the data
grid_search.fit(X, y)

# Print the best hyperparameters and best score
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)
