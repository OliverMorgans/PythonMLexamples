
# Importing the libraries
from sklearn.metrics import confusion_matrix
from keras.layers import Dense
from keras.models import Sequential
import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Part 2 -  Making the ANN

# Importing Keras libraries and packages


# initialising ANN
classifier = Sequential()
# first hidden layer
# relu kernel is a rectifier (0 then sloped to 1)
classifier.add(Dense(units=7, kernel_initializer='uniform',
                     activation='relu', input_dim=11))
# second hidden layer
classifier.add(Dense(units=7, kernel_initializer='uniform', activation='relu'))
# output layer
classifier.add(Dense(units=1, kernel_initializer='uniform',
                     activation='sigmoid', input_dim=11))
# compliling our ANN
classifier.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

# would use R-Squared for our metric to test this if we are doing a prediction

classifier.summary()


# fitting
classifier.fit(X_train, y_train, batch_size=10, epochs=50)

# Part 3 - Predicting
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_result = (y_pred > 0.5)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_result)
