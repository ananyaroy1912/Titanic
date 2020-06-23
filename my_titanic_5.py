#Created on Mon May  12 15:53:02 2020
#@author: Ananya Roy Choudhury


import pandas as pd
import numpy as np

dataset_train = pd.read_csv('train.csv')
dataset_test = pd.read_csv('test.csv')
y_train = dataset_train.iloc[:, 1].values
x_train = dataset_train.iloc[:, [2,4,5,6,7,9,10,11]].values

x_test2 = dataset_test.iloc[:, [1,3,4,5,6,8,9,10]].values

#Mean for empty columns
from sklearn.impute import SimpleImputer
#for numerical column
imputer = SimpleImputer(missing_values = np.nan,strategy = 'mean')
x_train[:,[2,5]] = imputer.fit_transform(x_train[:,[2,5]])
#for non numeric categorical
imputer1 = SimpleImputer(missing_values = np.nan,strategy = 'most_frequent')
x_train[:, [1,7]] = imputer1.fit_transform(x_train[:,[1,7]])
#for cabin special
imputer1 = SimpleImputer(missing_values = np.nan,strategy = 'constant',fill_value = '0')
x_train[:, [1,6]] = imputer1.fit_transform(x_train[:,[1,6]])

#Same as above but for test set
#for numerical column
imputer = SimpleImputer(missing_values = np.nan,strategy = 'mean')
x_test2[:,[2,5]] = imputer.fit_transform(x_test2[:,[2,5]])
#for non numeric categorical
imputer = SimpleImputer(missing_values = np.nan,strategy = 'most_frequent')
x_test2[:, [1,7]] = imputer.fit_transform(x_test2[:,[1,7]])
#for cabin special
imputer = SimpleImputer(missing_values = np.nan,strategy = 'constant',fill_value = '0')
x_test2[:, [1,6]] = imputer.fit_transform(x_test2[:,[1,6]])

#Analysis of dataframe
unique,counts = np.unique(x_test2[:,1],return_counts = True)

#Delete unnecesary variables 
del counts
del unique
del imputer
del imputer1
del label1
del label2
del label3
del ele
del i
del ct
del le_xtest
del le_xtrain

#For cabin lets code all values to 1 and no cabin to 0
for i,ele in enumerate(x_train[:,6]):
    if x_train[i,6] == '0':
        x_train[i,6] = 0 
    else:
       x_train[i,6] = 1    
       
for i,ele in enumerate(x_test2[:,6]):
    if x_test2[i,6] == '0':
        x_test2[i,6] = 0 
    else:
       x_test2[i,6] = 1 
unique,counts = np.unique(x_train[:,6],return_counts = True)

#Categorical
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
label1 = LabelEncoder()
x_train[:,1] = label1.fit_transform(x_train[:,1])
label2 = LabelEncoder()
x_train[:,7] = label2.fit_transform(x_train[:,7])
label3 = LabelEncoder()
x_train[:,0] = label3.fit_transform(x_train[:,0])
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [7])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                         # Leave the rest of the columns untouched
)
x_train = np.array(ct.fit_transform(x_train), dtype=np.float) 
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [3])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                         # Leave the rest of the columns untouched
)
x_train = np.array(ct.fit_transform(x_train), dtype=np.float) 
x_train = x_train[:, [1,2,4,5,6,7,8,9,10,11]]

#For test2
label1 = LabelEncoder()
x_test2[:,1] = label1.fit_transform(x_test2[:,1])
label2 = LabelEncoder()
x_test2[:,7] = label2.fit_transform(x_test2[:,7])
label3 = LabelEncoder()
x_test2[:,0] = label3.fit_transform(x_test2[:,0])
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [7])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                         # Leave the rest of the columns untouched
)
x_test2 = np.array(ct.fit_transform(x_test2), dtype=np.float)
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [3])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                         # Leave the rest of the columns untouched
)
x_test2 = np.array(ct.fit_transform(x_test2), dtype=np.float)
#To escape dummy variable trap
x_test2 = x_test2[:, [1,2,4,5,6,7,8,9,10,11]] 



#Feature scaling
from sklearn.preprocessing import StandardScaler
le_xtrain = StandardScaler()
x_train= le_xtrain.fit_transform(x_train)
#For test set
le_xtest = StandardScaler()
x_test2= le_xtest.fit_transform(x_test2) 

#Split Data
from sklearn.model_selection import train_test_split
x_train1,x_test1,y_train1,y_test1 = train_test_split(x_train,y_train,test_size = 0.10, random_state = 0) 


#Classifier
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 10))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(x_train1, y_train1, batch_size = 10, nb_epoch = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred1 = classifier.predict(x_test1)
y_pred1 = (y_pred1 > 0.5)


y_pred2 = classifier.predict(x_test2)
y_pred2 = (y_pred2 > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test1,y_pred1)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = x_train,y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

#Final Result
y_final = dataset_test.iloc[:, 0].values 
y_final = pd.Series(y_final)
y_pred2 = pd.Series(y_pred2)
y_final1 = np.array([y_final,y_pred2])
y_final1 = np.transpose(y_final1)
df = pd.DataFrame(y_final1)
df.to_excel (r'C:\Users\siddhartha_bhadra\Desktop\ARC_PY\Titanic\titanic output2.xlsx', index = False, header=True)

