#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 12:04:30 2018

@author: home
"""
#importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing Dataset
dataset=pd.read_csv('Salary_Data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values


#Spliting dataset into training set and test set
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.3,random_state=0)

#No need for Feature Scaling bcz ML model takes care for it but some models dont

#Fitting simple Linear regression model to training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#Predicting Test set values
y_pred= regressor.predict(x_test)

#Visualising Training set results
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary Vs Experience')
plt.xlabel('Years of Exp')
plt.ylabel('Salary')

#Visualising Testing set results
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary Vs Experience')
plt.xlabel('Years of Exp')
plt.ylabel('Salary')
