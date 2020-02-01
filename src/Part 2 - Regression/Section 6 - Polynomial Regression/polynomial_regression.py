# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 19:50:37 2019
@author: wahid
"""

#numpy a math tools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



#Importing Dataset
dataset = pd.read_csv('Position_Salaries.csv')
#get the independent data
X = dataset.iloc[:,1:-1].values
#get the dependent data
y = dataset.iloc[:,2].values


#Fitting linear regression to dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)


#Fitting Polynomial regression to dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


#Visualizing the Linear regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'pink')
plt.title('Linear Regression')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid),1)

#Visualize the Polynomial regression results
plt.scatter(X, y, color = 'red')
""" we will recalculate the X_poly incase the X has been modified before this line"""
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Ploynomial Regression')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()