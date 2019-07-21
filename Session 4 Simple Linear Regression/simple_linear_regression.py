# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 17:05:33 2019

@author: atulsingh01

We will train model basis on linear regression

We will pridict salary basis on exprience in year

Will match pridicted salary with actual
"""

# importing the ML libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# create dataset and read data file
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
# (:, :-1) means take all the line ,take all the col except last
Y = dataset.iloc[:, 1].values

#Split data set to training set and test set

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test =train_test_split(X,Y,test_size =1/3, random_state=0)


# Fitting simple linear regression to the training set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, Y_train, sample_weight=None)

# Pridict the test set results and save pridiction in vector

Y_pridict = regressor.predict(X_test) 

#Visualize the Training set results
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience Training set')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualize the Test set results
plt.scatter(X_test, Y_test, color='green')
# No need to change regressor on X_train becaz we already got the result
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience Pridiction set')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


















