#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier, \
    RandomForestClassifier
from xgboost.sklearn import XGBClassifier
import threading
from sklearn.grid_search import GridSearchCV
from sklearn.utils import resample
from sklearn.preprocessing import binarize

data = pd.read_csv(r"C:\Users\Smarta\Downloads\aps_failure_training_set_processed_8bit.csv")

x = data.iloc[:, 1:]
y = data.iloc[:, 0]

y[y > 0], y[y < 0] = 1, 0

skf = StratifiedKFold(n_splits=5)
predicted = np.array([np.zeros(y.shape[0]) for _ in range(3)])

#
# errors=[]
# states=[]
#
# rfc = GradientBoostingClassifier()
#
# param_grid = {
#    'n_estimators': [35,100,700],
#    'max_features': ['auto', 'log2'],
# }
#
# CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5, n_jobs=8)
# CV_rfc.fit(x, y)
# print(CV_rfc.best_params_)
#

def fitPredict(train, test):
    x_train = x.iloc[train, :]
    x_test = x.iloc[test, :]
    y_train = y[train]

    gradient_boosting = GradientBoostingClassifier(max_depth=4,
            learning_rate=0.08, min_samples_leaf=2)
    gradient_boosting.fit(x_train, y_train)
    predicted[0][test] = gradient_boosting.predict(x_test)

    xgboost = XGBClassifier()
    xgboost.fit(x_train, y_train)
    predicted[1][test] = xgboost.predict(x_test)

    random_forest = RandomForestClassifier(n_jobs=-1,
            max_features='auto', n_estimators=100, oob_score=True)
    random_forest.fit(x_train, y_train)
    predicted[2][test] = random_forest.predict(x_test)


threads = []

for (train, test) in skf.split(x, y):
    t = threading.Thread(target=fitPredict, args=(train, test))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

conf = []
for i in range(3):
    print(accuracy_score(y, predicted[i]))
    conf.append(confusion_matrix(y, predicted[i])) 

# weighted stacking

w1, w2, w3 = 1, 1, 2
weightsHalf = (w1 + w2 + w3) / 2

summedVotes = w1 * predicted[0] + w2 * predicted[1] + w3 * predicted[2]
netMajorityVotes = np.zeros(y.shape[0])
netMajorityVotes[summedVotes >= weightsHalf],\
netMajorityVotes[summedVotes < weightsHalf] = 1, 0
error = accuracy_score(y, netMajorityVotes)
print ('Overall accuracy: ', error)
