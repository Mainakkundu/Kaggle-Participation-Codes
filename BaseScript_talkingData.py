# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 11:14:15 2018

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
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.cross_validation import train_test_split
import lightgbm as lgb
import gc


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
print(train.shape)
print(test.shape)

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

print(train.columns)
print(test.columns)
print (train.head(2))
print (test.head(2))
print ('------------train & test data uploaded done---------')

	
from sklearn.utils import resample


# Separate majority and minority classes
df_majority = train[train['is_attributed']==0]
df_minority = train[train['is_attributed']==1]

df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=60000,    # to match majority class
                                 random_state=123) # reproducible results


# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
print(df_upsampled['is_attributed'].value_counts())

# Now because of memory issue just select a portion of data 
df_sam = df_upsampled.sample(n=80000)
print('----After random sampling on training data------')
print(df_sam['is_attributed'].value_counts())


X = df_sam[['ip','app','device','os','channel','dow','doy']]
X_test = test[['ip','app','device','os','channel','dow','doy']]
y = df_sam['is_attributed']
print('----- Upsampling Done and sucessfully extracted X,y --------')

# Create the training and test sets
X_train, X_val, y_train, y_val= train_test_split(X,y,test_size=0.2, random_state=123)
print('X_train,X_val...creation')
print(X_train.shape)
print(X_val.shape)
print(X_test.shape)
print(y_train.shape)
print(y_val.shape)

# Instantiate the XGBClassifier: xg_cl
print('--Very basic Xgboost Training start--')
xg_cl = xgb.XGBClassifier(n_estimators=100, objective='binary:logistic', seed=123)
xg_cl.fit(X_train, y_train)
preds_val = xg_cl.predict(X_val)
print(confusion_matrix(preds_val,y_val))
preds_test = xg_cl.predict(X_test) ## on unseen data set 

print('=====Base Training & testing done=======')

# generate submission datasets
my_submission = pd.DataFrame({'click_id': test['click_id'], 'is_attributed': preds_test})
my_submission.to_csv('submission_rf.csv', index=False)

print('@@@ WOW you sucessfully right your 1st kaggle script which gives a submission csv==== ')


