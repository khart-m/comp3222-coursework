import random
import numpy as np
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix, log_loss, roc_auc_score, precision_score, matthews_corrcoef, f1_score
from scipy.stats import wilcoxon

from solution.DecisionStumpClassifier import DecisionStumpClassifier
from solution.TreeEnsembleClassifier import TreeEnsembleClassifier
from provided_code import data_loaders as dl

'''
RQ1
'''

def cross_validation(file):
  X,y = dl.load_tabular_xy(file)
  pass

def metrics_ohe(file, metric, random_state):
  print("Testing normal data vs ohe for", file)
  X,y = dl.load_tabular_xy(file)
  ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
  X_ohe = ohe.fit_transform(X)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
  X_train_ohe, X_test_ohe, y_train_ohe, y_test_ohe = train_test_split(X_ohe, y, test_size=0.3, random_state=random_state)
  clf0 = TreeEnsembleClassifier(average_probas=True)
  clf1 = TreeEnsembleClassifier(average_probas=True)
  clf0.fit(X_train, y_train)
  clf1.fit(X_train_ohe, y_train_ohe)
  pred0 = clf0.predict(X_test)
  pred1 = clf1.predict(X_test_ohe)
  prob0 = clf0.predict_proba(X_test)
  prob1 = clf1.predict_proba(X_test_ohe)

  if(metric == "accuracy"):
    acc0 = accuracy_score(y_test, pred0)
    acc1 = accuracy_score(y_test_ohe, pred1)
    return acc0, acc1
  if(metric == "balanced_accuracy"):
    bacc0 = balanced_accuracy_score(y_test, pred0)
    bacc1 = balanced_accuracy_score(y_test_ohe, pred1)
    return bacc0, bacc1
  if(metric == "log_loss"):
    ll0 = log_loss(y_test, prob0, labels=np.unique(y))
    ll1 = log_loss(y_test_ohe, prob1, labels=np.unique(y))
    return ll0, ll1
  if(metric =="roc auc"): # need to check if its two class or multiclass first
    if(len(np.unique(y)) == 2):
      roc0 = roc_auc_score(y_test, prob0[:,1], labels=np.unique(y))
      roc1 = roc_auc_score(y_test_ohe, prob1[:,1], labels=np.unique(y))
    else:
      roc0 = roc_auc_score(y_test, prob0, labels=np.unique(y), multi_class="ovr")
      roc1 = roc_auc_score(y_test_ohe, prob1, labels=np.unique(y), multi_class="ovr")
    return roc0, roc1

def save_results(metrics, clf1_scores, clf2_scores, clf1_label, clf2_label, title, filename):

  x = np.arange(len(metrics))
  width = 0.35
  plt.figure(figsize = (10,6))
  bars1 = plt.bar(x - width/2, clf1_scores, width, label=clf1_label, color="mediumpurple")
  bars2 = plt.bar(x + width/2, clf2_scores, width, label=clf2_label, color="plum")

  plt.yticks(np.arange(0, 1.21, 0.2))
  plt.grid(axis='y', linestyle='--', alpha=0.6)

  def add_labels(bars):
    for bar in bars:
      height = bar.get_height()
      plt.text(
          bar.get_x() + bar.get_width() / 2,
          height + 0.01,
          f"{height:.2f}",
          ha="center",
          va="bottom",
          fontsize=9
      )
  add_labels(bars1)
  add_labels(bars2)

  plt.ylabel("Score")
  plt.ylim(0,1.2)
  plt.xticks(x, metrics)
  plt.title(title)
  plt.legend()

  plt.tight_layout()
  plt.savefig("rq1metrics.png")
  plt.show()

def test_fig():
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
  normalAcc = []
  oheAcc = []
  normalBacc = []
  oheBacc = []
  normalLL = []
  oheLL = []
  normalRoc = []
  oheRoc = []
  for file in files:
    nAcc, oAcc = metrics_ohe(file, "accuracy", 0)
    normalAcc.append(nAcc)
    oheAcc.append(oAcc)
    nBacc, oBacc = metrics_ohe(file, "balanced_accuracy", 0)
    normalBacc.append(nBacc)
    oheBacc.append(oBacc)
    nLL, oLL = metrics_ohe(file, "log_loss", 0)
    normalLL.append(nLL)
    oheLL.append(oLL)
    nR, oR = metrics_ohe(file, "roc auc", 0)
    normalRoc.append(nR)
    oheRoc.append(oR)



  normalAccuracy = np.average(normalAcc)
  oheAccuracy = np.average(oheAcc)
  normalBalancedAcc = np.average(normalBacc)
  oheBalancedAcc = np.average(oheBacc)
  normalLogLoss = np.average(normalLL)
  oheLogLoss = np.average(oheLL)
  normalRocAuc = np.average(normalRoc)
  oheRocAuc = np.average(oheRoc)
  normalMetrics = [normalAccuracy, normalBalancedAcc, normalLogLoss, normalRocAuc]
  oheMetrics = [oheAccuracy, oheBalancedAcc, oheLogLoss, oheRocAuc]
  print(normalRocAuc, oheRocAuc)

  metrics = ["Accuracy", "Balanced accuracy", "Log loss", "Roc auc"]
  save_results(metrics, normalMetrics, oheMetrics, "Normal", "OHE", "normal vs ohe", "normalVsOhe")

def test_wilcoxon():
  normal = [0.592809688688346,
0.6416666666666666,
0.09819321185876687,
0.8439162590902377,
0.3929195296734648,
0.3333333333333333,
0.5082101806239737,
0.6018756189344424,
0.5393285137634983,
0.41721622897824273,
0.8914170843776107,
0.366560693072321,
0.7655919021969376,
0.5,
0.9233862890270345,
0.6221231843142421,
0.7437141099046507,
0.6419136385532496,
0.4744416814513007,
0.6526753337182648,
0.5,
0.3595238095238095,]
  ohe = [0.5391057761436894,
0.7,
0.07018477592499961,
0.865264471485139,
0.3333333333333333,
0.3333333333333333,
0.5,
0.4864602663132075,
0.542419395190293,
0.37637009189640774,
0.8059492481203009,
0.3333333333333333,
0.763081862036295,
0.5,
0.9165089846045493,
0.5736700133196997,
0.7284187430241925,
0.5427585372353139,
0.4207248385125899,
0.678003476453042,
0.5,
0.35238095238095235,]
  print(normal, ohe)
  print(wilcoxon(normal, ohe, alternative="two-sided"))

def bacc_per_data():
  """
  Averages the balanced accuracy per dataset over random seeds 0 - 4
  Returns
  -------
  rows - datasets
  columns - bacc normal vs ohe data
  """
  output = []
  files = ["data/balance-scale/balance-scale.data",
           "data/balloons/balloons.data",
           "data/chess-krvk/chess-krvk.data",
           "data/chess-krvkp/chess-krvkp.data",
           "data/contraceptive-method/contraceptive-method.data",
           "data/connect-4/connect-4.data",
           "data/habermans-survival/habermans-survival.data",
           "data/hayes-roth/hayes-roth.data",
           "data/led-display/led-display.data",
           "data/lymphography/lymphography.data",
           "data/molecular-promoters/molecular-promoters.data",
           "data/molecular-splice/molecular-splice.data",
           "data/monks-1/monks-1.data",
           "data/monks-2/monks-2.data",
           "data/monks-3/monks-3.data",
           "data/nursery/nursery.data",
           "data/optdigits/optdigits.data",
           "data/pendigits/pendigits.data",
           "data/semeion/semeion.data",
           "data/spect-heart/spect-heart.data",
           "data/tic-tac-toe/tic-tac-toe.data",
           "data/zoo/zoo.data"]
  for f in files:
    nB = []
    oB = []
    for i in range(5):
      print(i)
      n, o = metrics_ohe(f, "balanced_accuracy", i)
      nB.append(n)
      oB.append(o)
    avNB = np.average(nB)
    avOB = np.average(oB)
    out = [avNB,avOB]
    output.append(out)
  return output

def test_ranked_acc():
  scores = bacc_per_data()
  winners = np.argmax(scores, axis=1)
  fileNames = ["balance-scale",
               "balloons",
               "chess-krvk",
               "chess-krvkp",
               "connect-4",
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
               "nursery",
               "optdigits",
               "pendigits",
               "semeion",
               "spect-heart",
               "tic-tac-toe",
               "zoo"]
  dataNames = ["Normal", "OHE"]
  fig, ax = plt.subplots(figsize = (8,3))
  ax.axis("off")

  table_data = []
  for i, fn in enumerate(fileNames):
    row = [fn] + list(scores[i]) + [winners[i]]
    table_data.append(row)
  columns = ['Dataset'] + dataNames + ['Winner']
  tbl = ax.table(cellText=table_data, colLabels=columns, loc='center')
  tbl.auto_set_font_size(False)
  tbl.set_fontsize(10)
  tbl.auto_set_column_width(col=list(range(len(columns))))
  df = pd.DataFrame(
      data=np.hstack([scores, np.array([winners]).T]),
      columns=dataNames + ['Winner']
  )
  df.insert(0, 'Dataset', fileNames)
  df.to_csv("rq1_bacc_table.csv", index=False)

  plt.show()
  return

