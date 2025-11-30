import random

import numpy as np
import pytest
import numpy as np
import pytest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix, log_loss, roc_auc_score, precision_score, matthews_corrcoef, f1_score

from solution.DecisionStumpClassifier import DecisionStumpClassifier
from solution.TreeEnsembleClassifier import TreeEnsembleClassifier
from provided_code import data_loaders as dl

'''
RQ1 Does TreeEnsembleClassifier perform better on average when handling categorical 
features natively than when used with one-hot encoding?

Use the provided datasets w/ categorical features

Choose one dataset for a deeper case study

Explore effect of key hyper parameters?

Compare classifiers using accuracy (or error) and any other justified metrics from 
the evaluation lecture (balanced accuracy and AUROC)

Sound experimental design (repeated resampling or cross validation)

CD diagram

aeon?
'''

# def train_test_split(file):
#   X,y = dl.load_tabular_xy(file)
#   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#   return X_train, X_test, y_train, y_test

def cross_validation(file):
  X,y = dl.load_tabular_xy(file)
  pass

def test_ohe():
  files = ["data/balance-scale/balance-scale.data",
           "data/balloons/balloons.data",
           "data/chess-krvkp/chess-krvkp.data",
           "data/contraceptive-method/contraceptive-method.data",
           "data/habermans-survival/habermans-survival.data",
           "data/hayes-roth/hayes-roth.data",
           "data/led-display/led-display.data",
           "data/lymphography/lymphography.data",
           "data/molecular-promoters/molecular-promoters.data",
           "data/molecular-splice/molecular-splice.data",
           "data/monks-1/monks-1.data",
           "data/monks-2/monks-2.data",
           "data/monks-3/monks-3.data",
           "data/optdigits/optdigits.data",
           "data/semeion/semeion.data",
           "data/spect-heart/spect-heart.data",
           "data/tic-tac-toe/tic-tac-toe.data",
           "data/zoo/zoo.data"]
  fileNames= ["balance-scale",
           "balloons",
           "chess-krvkp",
           "contraceptive-method",
           "habermans-survival",
           "hayes-roth",
           "led-display",
           "lymphography",
           "molecular-promoters",
           "molecular-splice",
           "monks-1",
           "monks-2",
           "monks-3",
           "optdigits",
           "semeion",
           "spect-heart",
           "tic-tac-toe",
           "zoo"]
  i=0
  for file in files:
    print("Testing normal data vs ohe for", fileNames[i])
    i += 1
    X,y = dl.load_tabular_xy(file)
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    X_train_ohe = ohe.fit_transform(X_train)
    X_test_ohe = ohe.transform(X_test)

    clf0 = TreeEnsembleClassifier(average_probas=False)
    clf1 = TreeEnsembleClassifier(average_probas=False)
    clf0.fit(X_train, y_train)
    clf1.fit(X_train_ohe, y_train)
    pred0 = clf0.predict(X_test)
    pred1 = clf1.predict(X_test_ohe)
    prob0 = clf0.predict_proba(X_test)
    prob1 = clf1.predict_proba(X_test_ohe)
    acc0 = accuracy_score(y_test, pred0)
    acc1 = accuracy_score(y_test, pred1)
    print("Accuracy:")
    print("normal: ", acc0)
    print("ohe: ", acc1)
    bacc0 = balanced_accuracy_score(y_test, pred0)
    bacc1 = balanced_accuracy_score(y_test, pred1)
    print("Balanced accuracy:")
    print("normal: ", bacc0)
    print("ohe: ", bacc1)
    cm0 = confusion_matrix(y_test, pred0)
    cm1 = confusion_matrix(y_test, pred1)
    print("Confusion matrix:")
    print("normal: \n", cm0)
    print("ohe: \n", cm1)
    #ll0 = log_loss(y_test, prob0,labels=clf0.classes_)
    #ll1 = log_loss(y_test, prob1,labels=clf1.classes_)
    #print("Log loss:")
    #print("normal: ", ll0)
    #print("ohe: ", ll1)



  return pred0, pred1, prob0, prob1


