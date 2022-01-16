import os
import joblib
import numpy as np
import pandas as pd

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


test_set_size = 0.2
seed= 120

raw = pd.read_json("data/raw.json", lines=True)

columns = len(raw.columns)
attributes = columns - 1

target_variable = columns

X_raw = raw.iloc[:, 0:attributes]
y_raw = raw.iloc[:, attributes]

X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=test_set_size, stratify=y_raw, random_state=seed)

X_train.to_json('data/X_train.json', orient="records")
X_test.to_json('data/X_test.json', orient="records")
y_train.to_json('data/y_train.json', orient="records")
y_test.to_json('data/y_test.json', orient="records")

