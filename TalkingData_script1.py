# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 09:41:34 2018

@author: mkundu
"""

import gc
import time
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


train_columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
test_columns  = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id']
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }

train = pd.read_csv('../input/train.csv',skiprows=range(1,123903891), nrows=61000000, usecols=train_columns, dtype=dtypes)
test = pd.read_csv('../input/test.csv',usecols=test_columns, dtype=dtypes)
print ('--------Data Uplpaded-------------')

def timeFeatures(df):
    # Make some new features with click_time column
    df['datetime'] = pd.to_datetime(df['click_time'])
    df['dow']      = df['datetime'].dt.dayofweek
    df["doy"]      = df["datetime"].dt.dayofyear
    #df["dteom"]    = df["datetime"].dt.daysinmonth - df["datetime"].dt.day
    df.drop(['click_time', 'datetime'], axis=1, inplace=True)
    return df

train = timeFeatures(train)
test = timeFeatures(test)
print ('------timeFeature done------')

### Data prep
X = train[['app','os', 'channel']] 
y = train['is_attributed']
X_test = test[['app','os','channel']]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)
print (X_train.shape)
print (X_test.shape)
print (X_val.shape)
print (y_train.shape)



# hyerparameters grid to search within
param_grid = [{'bootstrap': [False, True],
         'n_estimators': [80,90, 100, 110, 130],
         'max_features': [0.6, 0.65, 0.7, 0.73, 0.7500000000000001, 0.78, 0.8],
         'min_samples_leaf': [10, 12, 14],
         'min_samples_split': [3, 5, 7]}]
        
        

# declare the classifier
random_forest_classifier = RandomForestClassifier()

grid_search = GridSearchCV(random_forest_classifier, param_grid, cv=5,scoring='neg_mean_squared_error', refit=True)

# fine-tune the hyperparameters
grid_search.fit(X_train,y_train)

# get the best model
final_model = grid_search.best_estimator_

final_predictions_val = final_model.predict(X_val)

print('--------Validation Score----------')
print(classification_report(y_val, final_predictions_val)

# predict using the test dataset
final_predictions = final_model.predict_proba(X_test)[:,1]

# generate submission datasets
my_submission = pd.DataFrame({'click_id': test['click_id'], 'is_attributed': final_predictions})
my_submission.to_csv('submission_rf.csv', index=False)
#----------------------------------------------------------
