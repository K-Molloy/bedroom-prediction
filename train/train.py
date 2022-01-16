from datetime import datetime 
import os
import logging
import sys


import joblib
import numpy as np
import pandas as pd

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)


n_folds = 5
scoring = 'accuracy'
seed = 120
n_jobs = -2

logger.info("Loading data")
X_train_df = pd.read_json("data/X_train.json")
y_train_df = pd.read_json("data/y_train.json")

logger.info("Cleaning Data")
X_train_df[['number_habitable_rooms', 'number_heated_rooms']] = X_train_df[['number_habitable_rooms', 'number_heated_rooms']].astype(str)
# # Compose pipeline for the numerical and categorical features (Block #1 of 2)
numeric_columns = X_train_df.select_dtypes(include=['int64','float64']).columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', StandardScaler())
])
categorical_columns = X_train_df.select_dtypes(include=['object','bool']).columns
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='NA')),
    ('onehot', OneHotEncoder(sparse=False, handle_unknown='ignore'))
])

# # Compose pipeline for the numerical and categorical features (Block #2 of 2)
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_columns),
    ('cat', categorical_transformer, categorical_columns)
])

model_columns = list(X_train_df.columns)
joblib.dump(model_columns, 'data/model-params.joblib')


# Display the shapes of the training datasets for final inspection
# fit then transform, so that the preprocessor saves the steps for the prediction
preprocessor.fit(X_train_df)

joblib.dump(preprocessor, 'data/preprocessor.joblib')
logger.info("Preprocessor Dumped")

X_train = preprocessor.transform(X_train_df)
y_train = y_train_df.values.ravel()
logger.info("X_train.shape: {} y_train.shape: {}".format(X_train.shape, y_train.shape))

# Extra Trees model

start_time_model = datetime.now()
logger.info(f"Starting Training {start_time_model}")

RF_model = KNeighborsClassifier()
RF_model.fit(X_train, y_train)

# we could compress, but this model isnt big enough to warrant it
joblib.dump(RF_model, "data/optimal_model.joblib", compress=3)
logger.info('Model Dumped')






