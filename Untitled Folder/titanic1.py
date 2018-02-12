#import libraries
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import pandas as pd
import numpy as np

from sklearn.preprocessing import Imputer
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier



#load data
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


PATH = "/home/mitchell/kaggleData/titanicData/"

trainData = pd.read_csv(PATH + "train.csv")
testData = pd.read_csv(PATH + "test.csv")


predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked']

Y_train = trainData['Survived']
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

model = AdaBoostClassifier() # RandomForestClassifier() #GradientBoostingClassifier() # LinearSVC()

scores = cross_val_score(model, X_train, Y_train, scoring='neg_mean_absolute_error')
print(scores)
print('Mean Absolute Error %2f' %(-1 * scores.mean()))


#fit model
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

model.fit(X_train, Y_train)

#make prediction
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

prediction = model.predict(X_test)





#ensembling
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++









#output csv
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


my_submission = pd.DataFrame({'PassengerId': testData.PassengerId, 'Survived': prediction})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)
