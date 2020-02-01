# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 21:29:41 2019
@author: wahid
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importing Dataset
dataset = pd.read_csv('50_Startups.csv')
#get the independent data
X = dataset.iloc[:,:-1].values
#get the dependent data
y = dataset.iloc[:,4].values

#Labling the categorical fields
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [3])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                         # Leave the rest of the columns untouched
)
X = np.array(ct.fit_transform(X), dtype = np.float)

#Avoiding the dummy variable trap!
X = X[:, 1:]


#Splitting the dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


#Prediction the Test results
y_pred = regressor.predict(X_test)


#Building the Optimal Model by Backward Elimination
#Adding ones column to the matrix
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)

#new OLS regressor
import statsmodels.formula.api as sm

X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()


YY = regressor_OLS.predict(X_test)