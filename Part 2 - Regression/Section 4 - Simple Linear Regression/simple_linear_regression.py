# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 17:00:36 2019
@author: wahid
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importing Dataset
dataset = pd.read_csv('Salary_Data.csv')
#get the features matrix
X = dataset.iloc[:,:-1].values
#get the dependent data
y = dataset.iloc[:,1].values


#Splitting the dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


#Fitting Simple Linear Regression to dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predecting the test results 
y_pred = regressor.predict(X_test)

#Visualizing the training set results
plt.scatter(X_train, y_train, color = 'red')
plt.scatter(X_test, y_test, color = 'green')
#plt.scatter(X_test, y_pred, color = 'blue')
plt.plot(X_train, regressor.predict(X_train), color = 'black')
plt.title = 'Experiance vs Salary'
plt.xlabel = 'Years of Experiance'
plt.ylabel = 'Salary'
plt.show()





