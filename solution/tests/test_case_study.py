import random
import numpy as np
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix, log_loss, roc_auc_score, precision_score, matthews_corrcoef, f1_score
from scipy.stats import wilcoxon

from solution.DecisionStumpClassifier import DecisionStumpClassifier
from solution.TreeEnsembleClassifier import TreeEnsembleClassifier
from provided_code import data_loaders as dl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare
from aeon.visualisation import plot_critical_difference

# Additionally, you should choose one dataset for a deeper case study (analyse the behaviours of the RQ1 and RQ2 classifiers on this dataset).
# You should choose one the larger ones. You can find information on all the data sets on the UCI pages.
# You may also explore the effect of key hyper-parameters (e.g., number of estimators).


def metrics_ohe(file, random_state):
  print("Testing normal data vs ohe for", file)
  X,y = dl.load_tabular_xy(file)
  ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
  X_ohe = ohe.fit_transform(X)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
  X_train_ohe, X_test_ohe, y_train_ohe, y_test_ohe = train_test_split(X_ohe, y, test_size=0.3, random_state=random_state)
  treeN = TreeEnsembleClassifier(average_probas=True)
  treeO = TreeEnsembleClassifier(average_probas=True)
  randN = RandomForestClassifier(n_estimators=50, random_state=random_state)
  randO = RandomForestClassifier(n_estimators=50, random_state=random_state)
  adaN = AdaBoostClassifier(n_estimators=50, random_state=random_state)
  adaO = AdaBoostClassifier(n_estimators=50, random_state=random_state)
  treeN.fit(X_train, y_train)
  treeO.fit(X_train_ohe, y_train_ohe)
  predTN = treeN.predict(X_test)
  predTO = treeO.predict(X_test_ohe)
  probTN = treeN.predict_proba(X_test)
  probTO = treeO.predict_proba(X_test_ohe)
  randN.fit(X_train, y_train)
  randO.fit(X_train_ohe, y_train_ohe)
  predRN = randN.predict(X_test)
  predRO = randO.predict(X_test_ohe)
  probRN = randN.predict_proba(X_test)
  probRO = randO.predict_proba(X_test_ohe)
  adaN.fit(X_train, y_train)
  adaO.fit(X_train_ohe, y_train_ohe)
  predAN = adaN.predict(X_test)
  predAO = adaO.predict(X_test_ohe)
  probAN = adaN.predict_proba(X_test)
  probAO = adaO.predict_proba(X_test_ohe)

  accTN = accuracy_score(y_test, predTN)
  accTO = accuracy_score(y_test_ohe, predTO)
  accRN = accuracy_score(y_test, predRN)
  accRO = accuracy_score(y_test_ohe, predRO)
  accAN = accuracy_score(y_test, predAN)
  accAO = accuracy_score(y_test_ohe, predAO)

  acc = [accTN, accTO, accRN, accRO, accAN, accAO]

  baccTN = balanced_accuracy_score(y_test, predTN)
  baccTO = balanced_accuracy_score(y_test_ohe, predTO)
  baccRN = balanced_accuracy_score(y_test, predRN)
  baccRO = balanced_accuracy_score(y_test_ohe, predRO)
  baccAN = balanced_accuracy_score(y_test, predAN)
  baccAO = balanced_accuracy_score(y_test_ohe, predAO)

  bacc = [baccTN, baccTO, baccRN, baccRO, baccAN, baccAO]

  llTN = log_loss(y_test, probTN, labels=np.unique(y))
  llTO = log_loss(y_test_ohe, probTO, labels=np.unique(y))
  llRN = log_loss(y_test, probRN, labels=np.unique(y))
  llRO = log_loss(y_test_ohe, probRO, labels=np.unique(y))
  llAN = log_loss(y_test, probAN, labels=np.unique(y))
  llAO = log_loss(y_test_ohe, probAO, labels=np.unique(y))

  ll = [llTN, llTO, llRN, llRO, llAN, llAO]

  rocTN = roc_auc_score(y_test, probTN, labels=np.unique(y), multi_class="ovr")
  rocTO = roc_auc_score(y_test_ohe, probTO, labels=np.unique(y), multi_class="ovr")
  rocRN = roc_auc_score(y_test, probRN, labels=np.unique(y), multi_class="ovr")
  rocRO = roc_auc_score(y_test_ohe, probRO, labels=np.unique(y), multi_class="ovr")
  rocAN = roc_auc_score(y_test, probAN, labels=np.unique(y), multi_class="ovr")
  rocAO = roc_auc_score(y_test_ohe, probAO, labels=np.unique(y), multi_class="ovr")

  roc = [rocTN, rocTO, rocRN, rocRO, rocAN, rocAO]

  return acc, bacc, ll, roc


def test_get_metrics():
    accuracy = []
    balanced_accuracy = []
    log_loss = []
    roc_auc = []
    for i in range(5):
      acc, bacc, ll, roc = metrics_ohe("data/optdigits/optdigits.data", i)
      accuracy.append(acc)
      balanced_accuracy.append(bacc)
      log_loss.append(ll)
      roc_auc.append(roc)
    accuracy = np.array(accuracy).mean(axis=0)
    balanced_accuracy = np.array(balanced_accuracy).mean(axis=0)
    log_loss = np.array(log_loss).mean(axis=0)
    roc_auc = np.array(roc_auc).mean(axis=0)
    output = [accuracy, balanced_accuracy, log_loss, roc_auc]
    columns = ["Accuracy", "Balanced Accuracy", "Log Loss", "ROC AUC"]
    rows = np.column_stack(output)

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv("optidigitsMetrics.csv", index=False)

