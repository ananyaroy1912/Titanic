# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 15:46:38 2020

@author: Ananya Roy Choudhury
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Import Data
dataset = pd.read_csv('train.csv')
x = dataset.iloc[:, [2,4,5,6,7,9]].values
y = dataset.iloc[:, 1].values

#Missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(x[:, [2,5]])
x[:, [2,5]] = imputer.transform(x[:, [2,5]])

#Categorical Data
from sklearn.preprocessing import LabelEncoder ,OneHotEncoder
from sklearn.compose import ColumnTransformer
le_x = LabelEncoder()
x[:, 0] = le_x.fit_transform(x[:, 0])
x[:, 1] = le_x.fit_transform(x[:, 1])
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [0,1])],  
    remainder='passthrough'  )                      
x = np.array(ct.fit_transform(x), dtype=np.float)

# splitting test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state= 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
s_scale = StandardScaler()
x_train[:, [5,8]] = s_scale.fit_transform(x_train[:, [5,8]])
x_test[:, [5,8]] = s_scale.transform(x_test[:, [5,8]])


#Classification
# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', gamma = 0.2 , random_state = 0)
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

#Grid search
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['poly'],'degree': [2,3,4,5,6,7],'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],'coef0' : [0.1,0.0,0.2,0.5,1]},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(x_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_