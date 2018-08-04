# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


#NO NEED TO SPLIT AS DATA IS SMALL, ALSO BCZ IT IS NEGOTIATION
"""
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""
#NO NEED OF FEATURE SCALING, ML MODEL DOES IT
# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Fitting Polynomial Regression to Dataset
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4) #Changed to 2 to 3 to 4. model improves 
x_poly=poly_reg.fit_transform(X)
lin_reg=LinearRegression()
lin_reg.fit(x_poly,y)

#Visualising Polynomial Regreesion result
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1) ##This is to get curve not lines
plt.scatter(X,y,color='red')
plt.plot(X_grid,lin_reg.predict(poly_reg.fit_transform(X_grid)))
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')

#PREDICT SLUES
lin_reg.predict(poly_reg.fit_transform(6.5))