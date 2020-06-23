# -*- coding: utf-8 -*-
"""
Created on Mon May  4 15:53:02 2020

@author: Ananya Roy Choudhury
"""


import pandas as pd
import numpy as np

dataset_train = pd.read_csv('train.csv')
dataset_test = pd.read_csv('test.csv')
y_train = dataset_train.iloc[:, 1].values
x_train = dataset_train.iloc[:, [2,4,5,6,7,9,10,11]].values

#Mean for empty columns
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan,strategy = 'mean')
x_train[:,[2,5]] = imputer.fit_transform(x_train[:,[2,5]])

#Analysis[]
dataset_train[['Survived','Sex']].groupby(['Sex'],as_index = 'False').mean()
dataset_train[['Survived','Embarked']].groupby(['Embarked']).mean()
dataset_train[['Survived','Pclass']].groupby(['Pclass']).mean()
dataset_train[['Survived','Parch']].groupby(['Parch']).mean()
a = pd.factorize(x_train[:, 1]) # Makes a tuple. dnt knw why??
x_train[:,1] = a[0]
a = pd.factorize(x_train[:, 7])
x_train[:,7] = a[0]
x_train = x_train[: ,[0,1,2,3,4,5,7]]

#Feature scaling
from sklearn.preprocessing import StandardScaler
le_xtrain = StandardScaler()
x_train[:, [2,5]] = le_xtrain.fit_transform(x_train[:, [2,5]])

#Split Data
from sklearn.model_selection import train_test_split
x_tt1,x_test1,y_tt1,y_test1 = train_test_split(x_train,y_train,test_size = 0.2, random_state = 0)


#Classifier
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', gamma = 0.2 , random_state = 0)
classifier.fit(x_tt1,y_tt1)

y_pred1 = classifier.predict(x_tt1)
y_pred2 = classifier.predict(x_test1)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test1,y_pred2)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = x_tt1,y = y_tt1, cv = 10)
accuracies.mean()
accuracies.std()

