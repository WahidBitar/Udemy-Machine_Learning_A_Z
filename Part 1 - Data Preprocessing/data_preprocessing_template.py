# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 23:48:27 2019
@author: wahid
"""
#numpy a math tools
import numpy as np
#matplotlib is essencial for ML 
import matplotlib.pyplot as plt
#pandas to import and manage datasets
import pandas as pd



#Importing Dataset
dataset = pd.read_csv('Data.csv')
#get the independent data
X = dataset.iloc[:,:-1].values
#get the dependent data
y = dataset.iloc[:,3].values


#Splitting the dataset into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


"""
#Feature Scaling : Standardisation
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)
"""




