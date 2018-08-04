import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

## FEATURE SCALING
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

## FITING LOGISTIC REGRESSION to the TRAINING SET
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

## PREDICT TEST SET
y_pred=classifier.predict(X_test)

## CONFUSION MATRIX to check the correctness of the predicted values
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
## 65 + 24 correct prediction and 8 + 3 are incoorect pred

## VISUALISING RESULTS
