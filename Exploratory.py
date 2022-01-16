# %%
import os 
from datetime import datetime 

import folium
from folium import plugins
import rioxarray as rxr
import earthpy as et
import earthpy.spatial as es

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# %%
test_set_size = 0.2
val_set_size = 0.25
n_folds = 5
scoring = 'accuracy'
seed = 120
n_jobs = 4

# %%
raw = pd.read_json("data/raw.json", lines=True)

# %%
raw.head()

# %%
raw.describe()

# %%
raw.dtypes

# %%



# %% [markdown]
# ### Check for any missing values
# (not that i think your ETL pipelines are bad ;))

# %%
raw.isnull().sum()

# %% [markdown]
# No missing data imputation required

# %%
raw.info(verbose=True)

# %%
columns = len(raw.columns)
attributes = columns - 1

target_variable = columns

# %% [markdown]
# ### Split into Attributes and Target Variable

# %%
X_raw = raw.iloc[:, 0:attributes]
y_raw = raw.iloc[:, attributes]

# %%
X_train_val, X_test_df, y_train_val, y_test_df = train_test_split(X_raw, y_raw, test_size=test_set_size, stratify=y_raw, random_state=seed)
X_train_df, X_validation_df, y_train_df, y_validation_df = train_test_split(X_train_val, y_train_val, test_size=val_set_size, stratify=y_train_val, random_state=seed)

# %% [markdown]
# Splitting the dataset into test, train and validation sets to perform various performance analysis
# - stratify is used to ensure that no class (bedrooms) is under-represented in the splits
# - the seed is set to make results reproducible
# 
# The dataset is split into 3 parts
# - train size is 60%
# - test is 20%
# - validation is 20%

# %%
# Histograms for each attribute before pre-processing
columns_to_scale = X_train_df.columns[X_train_df.dtypes == 'int64'].tolist()

# %%
scaler = preprocessing.StandardScaler()
X_train_df[columns_to_scale] = scaler.fit_transform(X_train_df[columns_to_scale])

# also do validation set
X_validation_df[columns_to_scale] = scaler.fit_transform(X_validation_df[columns_to_scale])

# %%
# # Compose pipeline for the numerical and categorical features (Block #1 of 2)
numeric_columns = X_train_df.select_dtypes(include=['int64','uint8','float64']).columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', StandardScaler())
])
categorical_columns = X_train_df.select_dtypes(include=['object','bool']).columns
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='NA')),
    ('onehot', OneHotEncoder(sparse=False, handle_unknown='ignore'))
])

# %%
# # Compose pipeline for the numerical and categorical features (Block #2 of 2)
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_columns),
    ('cat', categorical_transformer, categorical_columns)
])

# Display the shapes of the training datasets for final inspection
X_train = preprocessor.fit_transform(X_train_df)
y_train = y_train_df.ravel()
print("X_train.shape: {} y_train.shape: {}".format(X_train.shape, y_train.shape))

# Display the shapes of the training datasets for final inspection
X_validation = preprocessor.fit_transform(X_validation_df)
y_validation = y_validation_df.ravel()
print("X_validation.shape: {} y_validation.shape: {}".format(X_validation.shape, y_validation.shape))


# %%
# Set up Algorithms Spot-Checking Array
start_time_training = datetime.now()

train_models = []
train_results = []
train_model_names = []
train_metrics = []

train_models.append(('LDA', LinearDiscriminantAnalysis()))
train_models.append(('CART', DecisionTreeClassifier(random_state=seed)))
train_models.append(('KNN', KNeighborsClassifier(n_jobs=n_jobs)))
train_models.append(('BGT', BaggingClassifier(random_state=seed, n_jobs=n_jobs)))
train_models.append(('RNF', RandomForestClassifier(random_state=seed, n_jobs=n_jobs)))
train_models.append(('EXT', ExtraTreesClassifier(random_state=seed, n_jobs=n_jobs)))

# %%
# Generate model in turn
for name, model in train_models:
    start_time_model = datetime.now()
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring, n_jobs=n_jobs, verbose=1)
    train_results.append(cv_results)
    train_model_names.append(name)
    train_metrics.append(cv_results.mean())
    print("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))
    print(model)
    print ('Model training time:', (datetime.now() - start_time_model), '\n')
print ('Average metrics ('+scoring+') from all models:',np.mean(train_metrics))
print ('Total training time for all models:',(datetime.now() - start_time_training))

# %% [markdown]
# ## Results Talk
# 
# The methods tested increase in training time and complexity (and also accuracy)
# 
# - LDA and CART are very simplistic yielding 0.63 (0.001) and 0.66 (0.0009) score respectively.
# - Improving on these KNN and BGT yield 0.73 (0.0004, 0.008), a respectable increase at the cost of training time.
# - RNF produces again increased, but with increased variablity 0.755 (0.001) but takes 9 minutes to train.
# - EXT adds more randomness, and leads to worse overall accuracy 0.752 (0.001) but requires half the time of RNF.

# %%
fig = plt.figure(figsize=(16,12))
fig.suptitle('Algorithm Comparison - Spot Checking')
ax = fig.add_subplot(111)
plt.boxplot(train_results)
ax.set_xticklabels(train_model_names)
plt.show()

# %%
n_jobs=1

# %%
# Set up the comparison array
tune_results = []
tune_model_names = []


# Tuning algorithm #1 - Extra Trees
start_time_model = datetime.now()

tune_model_1 = ExtraTreesClassifier(random_state=seed, n_jobs=n_jobs)
tune_model_names.append('EXT')
param_grid_1 = dict(n_estimators=np.array([10, 50, 100, 150, 200]))

kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
grid_1 = GridSearchCV(estimator=tune_model_1, param_grid=param_grid_1, scoring=scoring, cv=kfold, n_jobs=n_jobs, verbose=1)
grid_result_1 = grid_1.fit(X_train, y_train)

print("Best: %f using %s" % (grid_result_1.best_score_, grid_result_1.best_params_))
tune_results.append(grid_result_1.cv_results_['mean_test_score'])
means = grid_result_1.cv_results_['mean_test_score']
stds = grid_result_1.cv_results_['std_test_score']
params = grid_result_1.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
print ('Model training time:',(datetime.now() - start_time_model))

# %%
best_param_key_1 = list(grid_result_1.best_params_.keys())[0]
best_param_value_1 = list(grid_result_1.best_params_.values())[0]
print("Captured the best parameter for algorithm #1:", best_param_key_1, '=', best_param_value_1)

# %%
# Tuning algorithm #1 - Extra Trees
start_time_model = datetime.now()

tune_model_2 = RandomForestClassifier(random_state=seed, n_jobs=n_jobs)
tune_model_names.append('RNF')
param_grid_2 = dict(n_estimators=np.array([10, 50, 100, 150, 200]))

kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
grid_2 = GridSearchCV(estimator=tune_model_2, param_grid=param_grid_2, scoring=scoring, cv=kfold, n_jobs=n_jobs, verbose=1)
grid_result_2 = grid_2.fit(X_train, y_train)

print("Best: %f using %s" % (grid_result_2.best_score_, grid_result_2.best_params_))
tune_results.append(grid_result_2.cv_results_['mean_test_score'])
means = grid_result_2.cv_results_['mean_test_score']
stds = grid_result_2.cv_results_['std_test_score']
params = grid_result_2.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
print ('Model training time:',(datetime.now() - start_time_model))

# %%
best_param_key_2 = list(grid_result_2.best_params_.keys())[0]
best_param_value_2 = list(grid_result_2.best_params_.values())[0]
print("Captured the best parameter for algorithm #1:", best_param_key_2, '=', best_param_value_2)

# %%
fig = plt.figure(figsize=(16,12))
fig.suptitle('Algorithm Comparison - Post Tuning')
ax = fig.add_subplot(111)
plt.boxplot(tune_results)
ax.set_xticklabels(tune_model_names)
plt.show()

# %%
validation_model_1 = ExtraTreesClassifier(n_estimators=best_param_value_1, random_state=seed, n_jobs=n_jobs)
validation_model_1.fit(X_train, y_train)
print(validation_model_1)
predictions_1 = validation_model_1.predict(X_validation)
print('Accuracy Score:', accuracy_score(y_validation, predictions_1))
print(confusion_matrix(y_validation, predictions_1))
print(classification_report(y_validation, predictions_1))

# %%
validation_model_2 = RandomForestClassifier(n_estimators=best_param_value_2, random_state=seed, n_jobs=n_jobs)
validation_model_2.fit(X_train, y_train)
print(validation_model_2)
predictions_2 = validation_model_2.predict(X_validation)
print('Accuracy Score:', accuracy_score(y_validation, predictions_2))
print(confusion_matrix(y_validation, predictions_2))
print(classification_report(y_validation, predictions_2))

# %%
## Create the Final model

# %%
# Combining the training and validation datasets to form the complete dataset that will be used for training the final model
X_complete = np.vstack((X_train, X_validation))
y_complete = np.concatenate((y_train, y_validation))
print("X_complete.shape: {} y_complete.shape: {}".format(X_complete.shape, y_complete.shape))
test_model = validation_model_2.fit(X_complete, y_complete)
print(test_model)

# %%
# Apply feature scaling and transformation to the test dataset
scaled_features = scaler.transform(X_test_df[columns_to_scale])
X_test_df.loc[:,tuple(columns_to_scale)] = scaled_features
print(X_test_df.head())

# %%
# Finalize the test dataset for the modeling testing
X_test = X_test_df.to_numpy()
y_test = y_test_df.ravel()
print("X_test.shape: {} y_test.shape: {}".format(X_test.shape, y_test.shape))

# %%
test_predictions = test_model.predict(X_test)
print('Accuracy Score:', accuracy_score(y_test, test_predictions))
print(confusion_matrix(y_test, test_predictions))
print(classification_report(y_test, test_predictions))


