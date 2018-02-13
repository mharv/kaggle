#import libraries
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import pandas as pd
import numpy as np

from sklearn.preprocessing import Imputer
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
#from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

import keras
from keras.models import Sequential
from keras.layers import Dense


#load data
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


PATH = "../input/"

trainData = pd.read_csv(PATH + "train.csv")
testData = pd.read_csv(PATH + "test.csv")


predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked']

y_train = trainData['Survived']
X_train = trainData[predictors]
X_test = testData[predictors]


#preprocess
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#One-hot encoding first

encoded_training = pd.get_dummies(X_train)
encoded_test = pd.get_dummies(X_test)
X_train, X_test = encoded_training.align(encoded_test, join='left', axis=1)

#imputing after as some NaNs are generated from OHE

imputer = Imputer()

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

#cross validation
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

model = LinearSVC()

scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error')
print(scores)
print('Mean Absolute Error %2f' %(-1 * scores.mean()))

# feature scaling
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#new model
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Initialising the ANN
classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 346, kernel_initializer = 'uniform', activation = 'relu', input_dim = 691))
# Adding the second hidden layer
classifier.add(Dense(units = 346, kernel_initializer = 'uniform', activation = 'relu'))
# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))


# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
y_pred = y_pred.flatten(order='F')
y_pred = y_pred*1

#fit model
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#model.fit(X_train, Y_train)

#make prediction
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#prediction = model.predict(X_test)


#output csv
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


my_submission = pd.DataFrame({'PassengerId': testData.PassengerId, 'Survived': y_pred})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)
