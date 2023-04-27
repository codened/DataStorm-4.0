# %% [markdown]
# <a href="https://colab.research.google.com/github/codened/DataStorm-4.0/blob/main/stormingRound/DataStorm_4_0.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.dtreeg" alt="Open In Colab"/></a>

# %% [markdown]
# Path 
# stormingRound/DataStorm_4_0.ipynb

# %% [markdown]
# # Import necessary libraries

# %%
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# %% [markdown]
# # Importing Data Sets

# %%
rawHisTransDF=pd.read_csv('Historical-transaction-data.csv')
rawStoreInfDF=pd.read_csv('Store-info.csv')
rawTestDF=pd.read_csv('Testing-data.csv')

# %% [markdown]
# #### Viewing Dataframe

# %%
rawHisTransDF.head()

# %%
rawStoreInfDF.head()

# %% [markdown]
# # Data Pre Processing

# %% [markdown]
# ### Fixing Data

# %%
# convert the date string column to datetime
rawHisTransDF['transaction_date'] = pd.to_datetime(rawHisTransDF['transaction_date'], format='%Y/%m/%d').dt.date

# %%
# Performing left join
merged_df = pd.merge(rawHisTransDF, rawStoreInfDF, on='shop_id', how='left')

# %%
rawHisTransDF.describe(include='all').T

# %%
# get count of null values in each column
null_counts = merged_df.isnull().sum()
# print the counts
print(null_counts)

# %%
merged_df.dropna(subset=['item_description','invoice_id'], inplace=True)

# %%
# get count of null values in each column
null_counts = merged_df.isnull().sum()
# print the counts
print(null_counts)

# %%
merged_df.drop_duplicates(inplace=True)

# %% [markdown]
# ### Encoding 

# %%
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
merged_df['item_description'] = le.fit_transform(merged_df['item_description'])
merged_df['customer_id'] = le.fit_transform(merged_df['customer_id'])

# %%
merged_df['shop_id'] = merged_df['shop_id'].str.replace(r'^SHOP', '').astype(int)

# %%
merged_df['shop_profile'] = merged_df['shop_profile'].replace({'High': 1, 'Moderate': 2, 'Low': 3})
merged_df['shop_profile'] = merged_df['shop_profile'].fillna(0.0).astype(int)
merged_df['invoice_id'] = merged_df['invoice_id'].astype(int)

# %%
merged_df


# %%
print(merged_df[merged_df['quantity_sold'] == 0])

# %%
merged_df = merged_df[merged_df['quantity_sold'] != 0]

# %%
merged_df

# %% [markdown]
# # Feature Engineering

# %% [markdown]
# ### Feature Creation

# %%
merged_df['full_price'] = merged_df['quantity_sold'] * merged_df['item_price']

# %% [markdown]
# #### creating Avarage daily sales for each shop

# %%
merged_df['Daily_Sales'] = merged_df.groupby(['shop_id', 'transaction_date'])['full_price'].transform('sum')


# %%
merged_df

# %%
subset = merged_df.loc[(merged_df['transaction_date'] == pd.to_datetime('2021-12-11')) & (merged_df['shop_id'] == 8)]

# %%
# Group by shop id and calculate mean of daily_sales column
avg_sales = merged_df.groupby('shop_id')['Daily_Sales'].mean().reset_index()

# Merge the average sales data back into the original dataframe
merged_df = merged_df.merge(avg_sales, on='shop_id', suffixes=('', '_avg'))

# Print the updated dataframe
merged_df.head()

# %% [markdown]
# #### Full revinew

# %%
merged_df['revnew'] = merged_df.groupby(['shop_id'])['full_price'].transform('sum')

# %%
merged_df

# %% [markdown]
# #### Revnew per sqr feet of land

# %%
merged_df['rev_per_sqfeet'] = (merged_df['revnew'] / merged_df['shop_area_sq_ft']).round().astype(int)


# %%
merged_df

# %% [markdown]
# #### Avarage sold item types per each shop 

# %%
# group the original table by Shop ID and Transaction Date and count the unique items sold on each day
daily_items_sold = merged_df.groupby(['shop_id', 'transaction_date'])['item_description'].nunique().reset_index()

# group the resulting table by Shop ID and take the mean of the nunique column
avg_daily_items_sold = daily_items_sold.groupby('shop_id')['item_description'].mean().reset_index()

# rename the columns
avg_daily_items_sold.columns = ['shop_id', 'avd_daily_items_types_sold']
# convert float column to integers
avg_daily_items_sold['avd_daily_items_types_sold'] = avg_daily_items_sold['avd_daily_items_types_sold'].round().astype(int)

# merge with the original dataframe
merged_df = pd.merge(merged_df, avg_daily_items_sold, on='shop_id', how='left')

# %%
merged_df

# %% [markdown]
# #### Avarage Daily Transactions per each shop

# %%
# group the original table by Shop ID and Transaction Date and count the unique items sold on each day
daily_trans = merged_df.groupby(['shop_id', 'transaction_date'])['invoice_id'].nunique().reset_index()

# group the resulting table by Shop ID and take the mean of the nunique column
avg_daily_trans = daily_trans.groupby('shop_id')['invoice_id'].mean().reset_index()

# rename the columns
avg_daily_trans.columns = ['shop_id', 'avd_daily_transctions']
# convert float column to integers
avg_daily_trans['avd_daily_transctions'] = avg_daily_trans['avd_daily_transctions'].round().astype(int)

# merge with the original dataframe
merged_df = pd.merge(merged_df, avg_daily_trans, on='shop_id', how='left')

# %%
merged_df

# %% [markdown]
# #### Average number of custemers per day

# %%
# group the original table by Shop ID and Transaction Date and count the unique items sold on each day
daily_custemers = merged_df.groupby(['shop_id', 'transaction_date'])['customer_id'].nunique().reset_index()

# group the resulting table by Shop ID and take the mean of the nunique column
avg_daily_custemers = daily_custemers.groupby('shop_id')['customer_id'].mean().reset_index()

# rename the columns
avg_daily_custemers.columns = ['shop_id', 'avd_daily_custemers']
# convert float column to integers
avg_daily_custemers['avd_daily_custemers'] = avg_daily_custemers['avd_daily_custemers'].round().astype(int)

# merge with the original dataframe
merged_df = pd.merge(merged_df, avg_daily_custemers, on='shop_id', how='left')

# %%
merged_df

# %% [markdown]
# #### Persentage of Avarage number of time the same customer returning for the same shop

# %%
# calculate the number of times each customer visited each shop
visits = merged_df.groupby(['customer_id', 'shop_id'])['transaction_date'].count()
# calculate the average number of visits per customer per shop
avg_visits = visits.groupby(['shop_id']).mean()*100

avg_visits=avg_visits.round().astype(int)
# create a new DataFrame with the average visits
avg_visits_df = avg_visits.reset_index().rename(columns={'transaction_date': 'avg_visits'})

# merge the new DataFrame with the original DataFrame to add the average visits column
merged_df = pd.merge(merged_df, avg_visits_df, on=['shop_id'])

# %%
merged_df

# %% [markdown]
# # Visualizing

# %%
# Create correlation matrix
corr = merged_df.corr()

# Set figure size
plt.figure(figsize=(12, 8))

# Plot correlation matrix as heatmap
sns.heatmap(corr, annot=True, cmap='coolwarm')

# Display plot
plt.show()

# %%
# Drop highly co related features
cleanedDF = merged_df.drop(['avd_daily_custemers','transaction_date','revnew','item_price','item_description','quantity_sold','full_price','customer_id'], axis=1)

# %%
# drop duplicates
cleanedDF.drop_duplicates(inplace=True)

# %%
cleanedDF

# %%
# Create correlation matrix
corr = cleanedDF.corr()

# # Set figure size
# plt.figure(figsize=(12, 8))

# Plot correlation matrix as heatmap
sns.heatmap(corr, annot=True, cmap='coolwarm')

# Display plot
plt.show()

# %% [markdown]
# # Split To Test and Train Data

# %%
# Split the DataFrame into two based on column B
TestDF = cleanedDF[cleanedDF['shop_profile'] == 0].drop(['shop_profile'], axis=1)
TrainDF = cleanedDF[cleanedDF['shop_profile'] != 0]

# %%
# reset index
TestDF=TestDF.reset_index(drop=True)
TrainDF=TrainDF.reset_index(drop=True)

# %%
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt

# Separate the target variable
X = TrainDF.drop(['shop_profile'], axis=1)
y = TrainDF['shop_profile']

# Compute MI scores
mi_scores = mutual_info_classif(X, y)

# Convert to DataFrame and sort by MI score
mi_scores_df = pd.DataFrame({'feature': X.columns, 'mi_score': mi_scores})
mi_scores_df = mi_scores_df.sort_values('mi_score', ascending=False)

# Plot bar chart of MI scores
plt.figure(figsize=(12,8))
plt.bar(mi_scores_df['feature'], mi_scores_df['mi_score'])
plt.xticks(rotation=90)
plt.xlabel('Feature')
plt.ylabel('MI Score')
plt.show()

# %%
# Split Fulldata into training and testing sets
from sklearn.model_selection import train_test_split

column_name = 'shop_id'
unique_categories = TrainDF[column_name].nunique()
categories_in_dataset_1 = int(unique_categories * 0.6)
categories_in_dataset_2 = unique_categories - categories_in_dataset_1
dataset_1_categories = TrainDF[column_name].unique()[:categories_in_dataset_1]
dataset_2_categories = TrainDF[column_name].unique()[categories_in_dataset_1:]

train_data = TrainDF[TrainDF[column_name].isin(dataset_1_categories)]
test_data = TrainDF[TrainDF[column_name].isin(dataset_2_categories)]





#train_data, test_data = train_test_split(TrainDF, test_size=0.01)

# %%
test_data=test_data.reset_index(drop=True)
train_data=train_data.reset_index(drop=True)

# %%
# remove store id from the training and testing sets

train_data_noID = train_data.drop(['shop_id'], axis=1)
test_data_noID = test_data.drop(['shop_id'], axis=1)

# %% [markdown]
# # XG boost

# %%
train_data_noID['shop_profile'] = train_data_noID['shop_profile'].replace({1: 0, 2: 1, 3: 2})
test_data_noID['shop_profile'] = test_data_noID['shop_profile'].replace({1: 0, 2: 1, 3: 2})

# %%

# import xgboost as xgb
# from sklearn.model_selection import GridSearchCV, train_test_split

# # Split data into training and test sets
# X_train=train_data_noID.drop('shop_profile', axis=1)
# y_train=train_data_noID['shop_profile']
# X_test=test_data_noID.drop('shop_profile', axis=1)
# y_test=test_data_noID['shop_profile']

# # Set the parameters for grid search
# params = {
#     'n_estimators': [100, 500, 1000],
#     'learning_rate': [0.01, 0.1, 1],
#     'max_depth': [3, 5, 7],
#     'subsample': [0.5, 0.75, 1],
#     'colsample_bytree': [0.5, 0.75, 1],
#     'objective': ['multi:softmax', 'multi:softprob'],
#     'num_class': [3],
#     'tree_method': ['gpu_hist']
# }

# # Initialize the XGBoost classifier
# xgb_model = xgb.XGBClassifier()

# # Perform grid search to find the best hyperparameters
# grid_search = GridSearchCV(estimator=xgb_model, param_grid=params, scoring='accuracy', cv=3, n_jobs=-1, verbose=1)
# grid_search.fit(X_train, y_train)

# # Print the best hyperparameters found by grid search
# print(grid_search.best_params_)

# # Train the model using the best hyperparameters found by grid search
# xgb_model = xgb.XGBClassifier(**grid_search.best_params_)
# xgb_model.fit(X_train, y_train)

# # Make predictions on the test set
# xg_pred = xgb_model.predict(X_test)

# # Evaluate the model's performance on the test set
# accuracy = np.mean(xg_pred == y_test)
# print('Accuracy:', accuracy)



# %%


# %% [markdown]
# Best hyperparameters:  {'subsample': 1.0, 'reg_lambda': 0, 'reg_alpha': 0.1, 'n_estimators': 1000, 'max_depth': 7, 'learning_rate': 0.01, 'gamma': 0.5, 'colsample_bytree': 0.5}

# %%
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Split data into training and test sets
X_train=train_data_noID.drop('shop_profile', axis=1)
y_train=train_data_noID['shop_profile']
X_test=test_data_noID.drop('shop_profile', axis=1)
y_test=test_data_noID['shop_profile']


# Initialize the XGBoost classifier
xgb_model = xgb.XGBClassifier(tree_method='gpu_hist')

# fit model to training data
xgb_model.fit(X_train, y_train)

# make predictions on test data
xg_pred = xgb_model.predict(X_test)

# evaluate performance of model
mse = mean_squared_error(y_test, xg_pred)
print('MSE:', mse)

# %%
xg_pred=pd.DataFrame(xg_pred, columns=['shop_profile'])

train_data_noID['shop_profile'] = train_data_noID['shop_profile'].replace({0: 1, 1: 2, 2: 3})
test_data_noID['shop_profile'] = test_data_noID['shop_profile'].replace({0: 1, 1: 2, 2: 3})
xg_pred['shop_profile'] = xg_pred['shop_profile'].replace({0: 1, 1: 2, 2: 3})

# %%
# predicted_res = pd.concat([test_data['shop_id'], pred['shop_profile']], axis=1)
# expected_res=test_data[['shop_id', 'shop_profile']]

# pred_mode = predicted_res.groupby('shop_id')['shop_profile'].apply(lambda x: x.mode()[0]).reset_index()
# exp_mode = expected_res.groupby('shop_id')['shop_profile'].apply(lambda x: x.mode()[0]).reset_index()

# # import necessary libraries
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import f1_score

# y_test=exp_mode['shop_profile']
# y_pred=pred_mode['shop_profile']

# # calculate the F1 score for each class
# f1_class0 = f1_score(y_test, y_pred, labels=[1], average='weighted')
# f1_class1 = f1_score(y_test, y_pred, labels=[2], average='weighted')
# f1_class2 = f1_score(y_test, y_pred, labels=[3], average='weighted')

# # calculate the average F1 score
# f1_average = (f1_class0 + f1_class1 + f1_class2) / 3

# # print the results
# print(f"F1 score for class 0: {f1_class0:.2f}")
# print(f"F1 score for class 1: {f1_class1:.2f}")
# print(f"F1 score for class 2: {f1_class2:.2f}")
# print(f"Average F1 score: {f1_average:.2f}")


# %%
xg_pred

# %%
concatenated_df_XG_res = pd.concat([test_data['shop_id'], xg_pred['shop_profile']], axis=1)
# concatenated_df_XG_res['shop_profile'] = concatenated_df_XG_res['shop_profile'].astype(int)
# concatenated_df_XG_res['shop_id'] = concatenated_df_XG_res['shop_id'].astype(int)

# %%
concatenated_df_XG_res

# %%
expected_df_XG=test_data[['shop_id', 'shop_profile']]

# %%
expected_df_XG

# %%
# group by 'group' column and calculate mode of 'value' column
XG_res_mode_df = concatenated_df_XG_res.groupby('shop_id')['shop_profile'].apply(lambda x: x.mode()[0]).reset_index()

# %%
XG_res_mode_df

# %%
# group by 'group' column and calculate mode of 'value' column
XG_exp_mode_df = expected_df_XG.groupby('shop_id')['shop_profile'].apply(lambda x: x.mode()[0]).reset_index()

# %%
XG_exp_mode_df

# %%
# import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

y_test=XG_exp_mode_df['shop_profile']
y_pred=XG_res_mode_df['shop_profile']

# calculate the F1 score for each class
f1_class0 = f1_score(y_test, y_pred, labels=[1], average='weighted')
f1_class1 = f1_score(y_test, y_pred, labels=[2], average='weighted')
f1_class2 = f1_score(y_test, y_pred, labels=[3], average='weighted')

# calculate the average F1 score
f1_average = (f1_class0 + f1_class1 + f1_class2) / 3

# print the results
print(f"F1 score for class 0: {f1_class0:.2f}")
print(f"F1 score for class 1: {f1_class1:.2f}")
print(f"F1 score for class 2: {f1_class2:.2f}")
print(f"Average F1 score: {f1_average:.2f}")


# %%
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Assume y_true and y_pred are the true and predicted labels, respectively
cm = confusion_matrix(y_test, y_pred)

# Create a heatmap of the confusion matrix
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


# %% [markdown]
# # Random Forrest

# %%
# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Split data into training and test sets
X_train=train_data_noID.drop('shop_profile', axis=1)
y_train=train_data_noID['shop_profile']
X_test=test_data_noID.drop('shop_profile', axis=1)
y_test=test_data_noID['shop_profile']

# Initialize the Random Forest classifier
rfc = RandomForestClassifier(max_depth=50, min_samples_leaf=1, min_samples_split=2, n_estimators=25)

# Train the model on the training data
rfc.fit(X_train, y_train)

# Predict on the testing data
RF_pred = rfc.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, RF_pred)
print("Accuracy:", accuracy)



# %%
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GridSearchCV, train_test_split
# from sklearn.metrics import classification_report

# # Split data into training and test sets
# X_train=train_data_noID.drop('shop_profile', axis=1)
# y_train=train_data_noID['shop_profile']
# X_test=test_data_noID.drop('shop_profile', axis=1)
# y_test=test_data_noID['shop_profile']

# # Define the parameter grid to search over
# param_grid = {
#     'n_estimators': [50, 100, 150, 200],
#     'max_depth': [10, 20, 30, None],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': ['auto', 'sqrt']
# }

# # Create the Random Forest classifier
# rf = RandomForestClassifier(random_state=42)

# # Perform a grid search over the parameter grid with cross-validation
# rf_cv = GridSearchCV(rf, param_grid=param_grid, cv=5, n_jobs=-1)

# # Fit the grid search to the training data
# rf_cv.fit(X_train, y_train)

# # Print the best parameters found by the grid search
# print("Best Parameters:", rf_cv.best_params_)

# # Predict on the test data using the best model
# RF_pred = rf_cv.predict(X_test)

# # Print the classification report
# print(classification_report(y_test, RF_pred))


# %%
RF_pred = pd.DataFrame(RF_pred, columns=['shop_profile'])

predicted_res = pd.concat([test_data['shop_id'], RF_pred['shop_profile']], axis=1)
expected_res=test_data[['shop_id', 'shop_profile']]

pred_mode = predicted_res.groupby('shop_id')['shop_profile'].apply(lambda x: x.mode()[0]).reset_index()
exp_mode = expected_res.groupby('shop_id')['shop_profile'].apply(lambda x: x.mode()[0]).reset_index()

# import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

y_test=exp_mode['shop_profile']
y_pred=pred_mode['shop_profile']

# calculate the F1 score for each class
f1_class0 = f1_score(y_test, y_pred, labels=[1], average='weighted')
f1_class1 = f1_score(y_test, y_pred, labels=[2], average='weighted')
f1_class2 = f1_score(y_test, y_pred, labels=[3], average='weighted')

# calculate the average F1 score
f1_average = (f1_class0 + f1_class1 + f1_class2) / 3

# print the results
print(f"F1 score for class 0: {f1_class0:.2f}")
print(f"F1 score for class 1: {f1_class1:.2f}")
print(f"F1 score for class 2: {f1_class2:.2f}")
print(f"Average F1 score: {f1_average:.2f}")

# %%
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Assume y_true and y_pred are the true and predicted labels, respectively
cm = confusion_matrix(y_test, y_pred)

# Create a heatmap of the confusion matrix
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# %%
TestDF

# %%
X_test

# %%
Eval_noID=TestDF.drop('shop_id', axis=1)

# %%
Eval_noID

# %%
X_Eval=Eval_noID

# Predict on the evaluation set
RF_eval_pred = rfc.predict(X_Eval)

# %%
RF_eval_pred

# %%
RF_eval_pred = pd.DataFrame(RF_eval_pred, columns=['shop_profile'])

predicted_eval_res = pd.concat([TestDF['shop_id'], RF_eval_pred['shop_profile']], axis=1)


pred_Eval_mode = predicted_eval_res.groupby('shop_id')['shop_profile'].apply(lambda x: x.mode()[0]).reset_index()

# %%
pred_Eval_mode['shop_profile'] = pred_Eval_mode['shop_profile'].replace({1:'High', 2:'Moderate', 3:'Low'})

# %%
pred_Eval_mode

# %% [markdown]
# # Big Tune

# %%
train_data_noID['shop_profile'] = train_data_noID['shop_profile'].replace({1: 0, 2: 1, 3: 2})
test_data_noID['shop_profile'] = test_data_noID['shop_profile'].replace({1: 0, 2: 1, 3: 2})
test_data['shop_profile'] = test_data['shop_profile'].replace({1: 0, 2: 1, 3: 2})
train_data['shop_profile'] = train_data['shop_profile'].replace({1: 0, 2: 1, 3: 2})

# %%
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


# %%
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# Create a custom scoring function
def custom_scorer(y_true, y_pred):
    selected_testData = test_data.loc[y_pred.index]
    # predicted_res = pd.concat([test_data['shop_id'], y_pred['shop_profile']], axis=1)
    # merge dataframes on row index
    predicted_res = y_pred['shop_profile'].merge(test_data['shop_id'], left_index=True, right_index=True, how='left')
    expected_res=selected_testData[['shop_id', 'shop_profile']]

    # reset indexes
    predicted_res=predicted_res.reset_index(drop=True)
    expected_res=expected_res.reset_index(drop=True)
    
    predicted_res
    expected_res.head(10)
    
    pred_mode = predicted_res.groupby('shop_id')['shop_profile'].apply(lambda x: x.mode()[0]).reset_index()
    exp_mode = expected_res.groupby('shop_id')['shop_profile'].apply(lambda x: x.mode()[0]).reset_index()
    
    pred_mode.head(10)
    exp_mode.head(10)

    # import necessary libraries
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import f1_score

    y_test=exp_mode['shop_profile']
    y_pred=pred_mode['shop_profile']

    # calculate the F1 score for each class
    f1_class0 = f1_score(y_test, y_pred, labels=[1], average='weighted')
    f1_class1 = f1_score(y_test, y_pred, labels=[2], average='weighted')
    f1_class2 = f1_score(y_test, y_pred, labels=[3], average='weighted')

    # calculate the average F1 score
    f1_average = (f1_class0 + f1_class1 + f1_class2) / 3

    # print the results
    print(f"F1 score for class 0: {f1_class0:.2f}")
    print(f"F1 score for class 1: {f1_class1:.2f}")
    print(f"F1 score for class 2: {f1_class2:.2f}")
    print(f"Average F1 score: {f1_average:.2f}")
    
    return f1_average

# Split data into training and test sets
X_train=train_data_noID.drop('shop_profile', axis=1)
y_train=train_data_noID['shop_profile']
X_test=test_data_noID.drop('shop_profile', axis=1)
y_test=test_data_noID['shop_profile']

# Define the models to be hyperparameter tuned
models = [
    {
        'name': 'XGBoost',
        'model': XGBClassifier(tree_method='gpu_hist'),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7]
        }
    },
    {
        'name': 'Random Forest',
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7]
        }
    },
    {
        'name': 'Logistic Regression',
        'model': LogisticRegression(penalty='l2'),
        'params': {
            'C': [0.1, 1.0, 10.0]
        }
    },
    {
        'name': 'KNN',
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [3, 5, 7],
            'p': [1, 2]
        }
    }
]

# Perform hyperparameter tuning for each model using GridSearchCV
best_model = None
best_score = 0.0
for model_config in models:
    print(f'Tuning {model_config["name"]}...')
    model = model_config['model']
    params = model_config['params']
    custom_grid_search = GridSearchCV(
        model,
        params,
        cv=5,
        scoring=make_scorer(custom_scorer),
        n_jobs=-1
    )
    custom_grid_search.fit(X_train, y_train)
    score = custom_grid_search.best_score_
    print(f'Best score for {model_config["name"]}: {score:.4f}')
    print(f'Best Parametersfor {model_config["name"]}:  {custom_grid_search.best_params_}')
    if score > best_score:
        best_score = score
        best_model = custom_grid_search.best_estimator_
        best_model_name = model_config['name']
        best_model_hyperparams = custom_grid_search.best_params_

# Train the best model on the full training set
best_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Calculate accuracy score on the test set
accuracy = accuracy_score(y_test, y_pred)
print(f'Test accuracy score for the best model: {accuracy:.4f}')


# %%
train_data_noID['shop_profile'] = train_data_noID['shop_profile'].replace({0: 1, 1: 2, 2: 3})
test_data_noID['shop_profile'] = test_data_noID['shop_profile'].replace({0: 1, 1: 2, 2: 3})
test_data['shop_profile'] = test_data['shop_profile'].replace({0: 1, 1: 2, 2: 3})
train_data['shop_profile'] = train_data['shop_profile'].replace({0: 1, 1: 2, 2: 3})
y_pred['shop_profile'] = y_pred['shop_profile'].replace({0: 1, 1: 2, 2: 3})

# %%
y_pred

# %%
import pandas as pd

# create two dataframes
df1 = pd.DataFrame({'A1': [1, 2, 3,7], 'B1': ['a', 'b', 'c','g']}, index=[5, 3, 4,6])
df2 = pd.DataFrame({'A2': [4, 5, 6], 'B2': ['d', 'e', 'f']}, index=[3, 4, 5])

# concatenate them based on row index
concatenated_df = pd.concat([df1, df2], axis=1)

# merge dataframes on row index
result = df2.merge(df1, left_index=True, right_index=True, how='left')


# %%
concatenated_df

# %%
result


