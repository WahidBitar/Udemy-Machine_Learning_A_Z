# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 19:50:37 2019
@author: wahid
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importing Dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values


#Standardisation
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1,1))


#Fitting SVR to dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X,y)

y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid),1)

#Visualize the SVR results
plt.scatter(X, y, color = 'red')
""" we will recalculate the X_poly incase the X has been modified before this line"""
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('SVR')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()