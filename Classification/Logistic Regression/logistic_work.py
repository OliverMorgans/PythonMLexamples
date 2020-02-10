# Logistic Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4]


# splitting into training and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0 )


# Feature Scaling
from sklearn.preprocessing import StandardScaler as stan
sc_x = stan()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.fit_transform(X_test)

# Fitting logistic regression to the training set
from sklearn.linear_model import LogisticRegression as LR
classifier = LR(random_state = 0)
classifier.fit(X_train, y_train)

#prediction
y_pred = classifier.predict(X_test)

# Making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Visualising the training set
from matplotlib.colors import ListedColorMap
X_set, y_set = X_train, y_train
