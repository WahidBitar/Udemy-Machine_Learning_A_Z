# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 23:48:27 2019

@author: wahid
"""

#numpy is for a math tools
import numpy as np
#matplotlib is essencial from ML 
import matplotlib.pyplot as plt
#pandas to import and manage data sets
import pandas as pd


#get the data
dataset = pd.read_csv('Data.csv')

#get the independent data
X = dataset.iloc[:,:-1].values
#get the dependent data
y = dataset.iloc[:,3].values

#fix the missing data
#import sklearn.preprocessing
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan ,strategy='mean')
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

#Labling the categorical fields
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

#pivoting the first column to array
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [0])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                         # Leave the rest of the columns untouched
)
X = np.array(ct.fit_transform(X), dtype = np.float)


#labling result
lableencoder_y = LabelEncoder()
y = lableencoder_y.fit_transform(y)



#Prepare test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#Standardisation
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)





