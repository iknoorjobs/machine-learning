#importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing Dataset
dataset=pd.read_csv('Data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values

#Taking care of mising values
from sklearn.preprocessing import Imputer
imp=Imputer(missing_values="NaN", strategy="mean", axis=0)
imp=imp.fit(x[:,1:3])
x[:,1:3]=imp.transform(x[:,1:3])

#Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x=LabelEncoder()
x[:,0]=labelencoder_x.fit_transform(x[:,0])
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)
onehot=OneHotEncoder(categorical_features=[0])
x=onehot.fit_transform(x).toarray()

#Spliting dataset into training set and test set
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


