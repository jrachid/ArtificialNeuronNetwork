# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 14:30:36 2018

@author: Rachid
"""

### Data Preprocessing ###

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, -1].values

# Encoding categorical data - independent variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x1 = LabelEncoder()
X[:,1] = labelencoder_x1.fit_transform(X[:,1])

labelencoder_x2 = LabelEncoder()
X[:,2] = labelencoder_x2.fit_transform(X[:,2])

# Création des colonnes France/Spain/Germany
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

# Dummy variable
X = X[:,1:] 

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

### Build ANN ###

# Importing Keras modules
import keras
# Module for initialization of the ANN
from keras.models import Sequential
# Module for creating layers inside the ANN
from keras.layers import Dense, Dropout

# Initialization of the ANN
classifier = Sequential()

# Add enter layer and hidden layer
# Utilisation de la fonction redresseur dans le réseau et la fonction sigmoid pour la sortie
classifier.add(Dense(units = 6, activation = "relu", kernel_initializer = "uniform", input_dim=11))

# utilisation de la classe Dropout pour réduire l'overfitting
classifier.add(Dropout(rate=0.1))

# Add a second hidden layer
classifier.add(Dense(units = 6, activation = "relu", kernel_initializer = "uniform"))
classifier.add(Dropout(rate=0.1))

# Add the output layer ( with probability)
classifier.add(Dense(units = 1, activation = "sigmoid", kernel_initializer = "uniform"))

# compiler the ANN (with Stochastic Gradient Descent)
classifier.compile(optimizer = "adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the ANN
classifier.fit(X_train, y_train, batch_size = 10, epochs=100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
#♣ transform y_pred to boolean
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Make a prediction
classifier.predict(sc.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]]))) > 0.5

# Use cross_val_score to evaluate the ANN

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, activation = "relu", kernel_initializer = "uniform",
                         input_dim=11))
    classifier.add(Dense(units = 6, activation = "relu", kernel_initializer = "uniform"))
    classifier.add(Dense(units = 1, activation = "sigmoid", kernel_initializer = "uniform"))
    classifier.compile(optimizer = "adam", loss="binary_crossentropy", metrics=["accuracy"])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)

# Mesurer la précision avec la validation croisée
precisions = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)

moy_precisions= precisions.mean()
sd_precisions = precisions.std()

# Improve the ANN - search the best parameters
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

# the best parameters found are batch_size = 32 / epochs = 500 / optimizer = adam
# we train again the ANN with the best parameters
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = "adam", loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 32, epochs=500)
# train the ann
classifier.fit(X_train, y_train, batch_size = 32, epochs=500)

# Make a new prediction
input_data = np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]])
new_prediction = classifier.predict(sc.transform(input_data ))
print(new_prediction > 0.5)

# New prediction for Test set
y_pred_new = classifier.predict(X_test)
y_pred_new = (y_pred_new > 0.5)

# Making the New Confusion Matrix
cm_new = confusion_matrix(y_test, y_pred_new)

