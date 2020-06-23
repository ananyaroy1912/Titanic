#Created on Mon May  12 15:53:02 2020
#@author: Ananya Roy Choudhury


import pandas as pd
import numpy as np

dataset_train = pd.read_csv('train.csv')
dataset_test = pd.read_csv('test.csv')
y_train = dataset_train.iloc[:, 1]
x_train = dataset_train.iloc[:, 2:]

x_test2 = dataset_test.iloc[:, [1,3,4,5,6,7,8,9,10]]


x_train[['Parch','SibSp']].corr()

x_train['tckt_grp'] = x_train.groupby('Ticket').mean()
 
from sklearn.preprocessing import LabelEncoder
label_x = LabelEncoder()
x_train['tckt_grp'] = label_x.fit_transform(x_train['Ticket'])
x_train['tckt_grp'].max()
x_train[['tckt_grp','new_fare']].corr()
x_train.groupby('tckt_grp')['Fare']

x_train['cnt_tckt'] =  x_train.groupby('Ticket')['Fare'].transform('count')

x_train.drop('cnt_tckt', axis = 1, inplace = True)
x_train[x_train['Ticket'] == '250644'][['Fare','Ticket','tckt_grp','cnt_tckt','new_fare']]
x_train['new_fare'] = x_train['Fare']/x_train['cnt_tckt']
x_train.drop('cnt_tckt', axis = 1, inplace = True)
x_train.drop('Fare', axis = 1, inplace = True)
x_train.drop('Ticket', axis = 1, inplace = True)
x_train.drop('Name', axis = 1, inplace = True)
z.drop()

x_train['new_cab'] = x_train['Cabin'].transform(lambda x:x is not np.nan)
x_train[x_train['Cabin'] == np.nan][['Cabin','new_cab']]
x_train.drop('Cabin', axis = 1, inplace = True)
label_x = LabelEncoder()
x_train['Pclass'] = label_x.fit_transform(x_train['Pclass'])
label_x = LabelEncoder()
x_train['Sex'] = label_x.fit_transform(x_train['Sex'])
label_x = LabelEncoder()
x_train['Embarked'] = label_x.fit_transform(x_train['Embarked'])

#Missing Values
from sklearn.impute import SimpleImputer
x_imp = SimpleImputer(missing_values = np.nan,strategy = 'mean')
x_train[['Age','new_fare']] = x_imp.fit_transform(x_train[['Age','new_fare']])
x_imp = SimpleImputer(missing_values = np.nan,strategy = 'most_frequent')
x_train[['Embarked','Sex']] = x_imp.fit_transform(x_train[['Embarked','Sex']])

#One Hot encoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [0])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                         # Leave the rest of the columns untouched
)
x_train = np.array(ct.fit_transform(x_train), dtype=np.float)
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [6])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                         # Leave the rest of the columns untouched
)
x_train = np.array(ct.fit_transform(x_train), dtype=np.float)
x_train = x_train[:, [1,2,3,4,5,6,7,8,9]]

#Feature scaling
from sklearn.preprocessing import StandardScaler
x_scaler = StandardScaler()
#y_scaler = StandardScaler()
x_train[:,[5,8]] = x_scaler.fit_transform(x_train[:,[5,8]])
#y_train = y_scaler.fit_transform(np.array(y_train).reshape(-1,1))
#y_train = np.array(y_train).reshape(891,)

#Split Data
from sklearn.model_selection import train_test_split
x_train1,x_test1,y_train1,y_test1 = train_test_split(x_train,y_train,test_size = 0.20, random_state = 0)


#Classifier
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', gamma = 0.2 , random_state = 0)
classifier.fit(x_train1,y_train1)

y_pred1 = classifier.predict(x_test1)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test1,y_pred1)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = x_train1,y = y_train1, cv = 10)
accuracies.mean()
accuracies.std()