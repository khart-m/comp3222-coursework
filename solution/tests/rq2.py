import random
import numpy as np
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
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

'''
RQ2
'''

def metrics_(file, random_state):
  print("Testing normal data vs ohe for", file)
  X,y = dl.load_tabular_xy(file)
  ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
  X_ohe = ohe.fit_transform(X)
  X_train, X_test, y_train, y_test = train_test_split(X_ohe, y, test_size=0.3, random_state=random_state)
  clf0 = TreeEnsembleClassifier(average_probas=True, random_state=random_state)
  clf1 = RandomForestClassifier(n_estimators=50, random_state=random_state)
  clf2 = AdaBoostClassifier(n_estimators=50, random_state=random_state)
  clf0.fit(X_train, y_train)
  clf1.fit(X_train, y_train)
  clf2.fit(X_train, y_train)
  pred0 = clf0.predict(X_test)
  pred1 = clf1.predict(X_test)
  pred2 = clf2.predict(X_test)
  prob0 = clf0.predict_proba(X_test)
  prob1 = clf1.predict_proba(X_test)
  prob2 = clf2.predict_proba(X_test)

  acc0 = accuracy_score(y_test, pred0)
  acc1 = accuracy_score(y_test, pred1)
  acc2 = accuracy_score(y_test, pred2)
  a = [acc0, acc1, acc2]
  bacc0 = balanced_accuracy_score(y_test, pred0)
  bacc1 = balanced_accuracy_score(y_test, pred1)
  bacc2 = balanced_accuracy_score(y_test, pred2)
  b = [bacc0, bacc1, bacc2]
  ll0 = log_loss(y_test, prob0, labels=clf0.classes_)
  ll1 = log_loss(y_test, prob1, labels=clf1.classes_)
  ll2 = log_loss(y_test, prob2, labels=clf2.classes_)
  l= [ll0, ll1, ll2]
  if(len(np.unique(y)) == 2):
    roc0 = roc_auc_score(y_test, prob0[:,1], labels=np.unique(y))
    roc1 = roc_auc_score(y_test, prob1[:,1], labels=np.unique(y))
    roc2 = roc_auc_score(y_test, prob2[:,1], labels=np.unique(y))
  else:
    roc0 = roc_auc_score(y_test, prob0, labels=np.unique(y), multi_class="ovr")
    roc1 = roc_auc_score(y_test, prob1, labels=np.unique(y), multi_class="ovr")
    roc2 = roc_auc_score(y_test, prob2, labels=np.unique(y), multi_class="ovr")
  r = [roc0, roc1, roc2]
  return a, b, l, r

def save_results(metrics, clf1_scores, clf2_scores, clf3_scores, clf1_label, clf2_label, clf3_label, title, filename):
  x = np.arange(len(metrics))
  width = 0.25
  plt.figure(figsize = (10,6))
  bars1 = plt.bar(x - width, clf1_scores, width, label=clf1_label, color="mediumpurple")
  bars2 = plt.bar(x, clf2_scores, width, label=clf2_label, color="plum")
  bars3 = plt.bar(x + width, clf3_scores, width, label=clf3_label, color="lightsteelblue")

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
  add_labels(bars3)

  plt.ylabel("Score")
  plt.ylim(0,1.2)
  plt.xticks(x, metrics)
  plt.title(title)
  plt.legend()

  plt.tight_layout()
  plt.savefig("rq2metrics.png")
  plt.show()

def test_metrics():
  treeAcc = []
  forestAcc = []
  adaAcc = []
  treeBacc = []
  forestBacc = []
  adaBacc = []
  treeLL = []
  forestLL = []
  adaLL = []
  treeRA = []
  forestRA = []
  adaRA = []
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
    tAcc = []
    tBacc = []
    tLl = []
    tRc = []
    rAcc = []
    rBacc = []
    rLl = []
    rRc = []
    aAcc = []
    aBacc = []
    aLl = []
    aRc = []
    for i in range(5):
      a, b, l, r = metrics_(f, i)
      tAcc.append(a[0])
      tBacc.append(b[0])
      tLl.append(l[0])
      tRc.append(r[0])
      rAcc.append(a[1])
      rBacc.append(b[1])
      rLl.append(l[1])
      rRc.append(r[1])
      aAcc.append(a[2])
      aBacc.append(b[2])
      aLl.append(l[2])
      aRc.append(r[2])
    treeAcc.append(np.average(tAcc))
    treeBacc.append(np.average(tBacc))
    treeLL.append(np.average(tLl))
    treeRA.append(np.average(tRc))
    forestAcc.append(np.average(rAcc))
    forestBacc.append(np.average(rBacc))
    forestLL.append(np.average(rLl))
    forestRA.append(np.average(rRc))
    adaAcc.append(np.average(aAcc))
    adaBacc.append(np.average(aBacc))
    adaLL.append(np.average(aLl))
    adaRA.append(np.average(aRc))
  print(treeAcc)
  print(treeBacc)
  print(treeLL)
  print(treeRA)
  print(forestAcc)
  print(forestBacc)
  print(forestLL)
  print(forestRA)
  print(adaAcc)
  print(adaBacc)
  print(adaLL)
  print(adaRA)
  np.savetxt("treeBacc.csv", treeBacc, delimiter=",")
  np.savetxt("forestBacc.csv", forestBacc, delimiter=",")
  np.savetxt("adaBacc.csv", adaBacc, delimiter=",")
  tree_scores = [np.average(treeAcc), np.average(treeBacc), np.average(treeLL), np.average(treeRA)]
  forest_scores = [np.average(forestAcc), np.average(forestBacc), np.average(forestLL), np.average(forestRA)]
  ada_scores = [np.average(adaAcc), np.average(adaBacc), np.average(adaLL), np.average(adaRA)]
  metrics = ["Accuracy", "Balanced accuracy", "Log loss", "Roc auc"]
  save_results(metrics, tree_scores, forest_scores, ada_scores, "Tree Ensemble", "Random Forest", "AdaBoost", "treeEnsemble vs randomForest vs adaBoost", "rq2metrics")

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

def bacc(file, random_state):
  print("Testing normal data vs ohe for", file)
  X,y = dl.load_tabular_xy(file)
  ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
  X_ohe = ohe.fit_transform(X)
  X_train, X_test, y_train, y_test = train_test_split(X_ohe, y, test_size=0.3, random_state=random_state)
  clf0 = TreeEnsembleClassifier(average_probas=True, random_state=random_state)
  clf1 = RandomForestClassifier(n_estimators=50, random_state=random_state)
  clf2 = AdaBoostClassifier(n_estimators=50, random_state=random_state)
  clf0.fit(X_train, y_train)
  clf1.fit(X_train, y_train)
  clf2.fit(X_train, y_train)
  pred0 = clf0.predict(X_test)
  pred1 = clf1.predict(X_test)
  pred2 = clf2.predict(X_test)

  bacc0 = balanced_accuracy_score(y_test, pred0)
  bacc1 = balanced_accuracy_score(y_test, pred1)
  bacc2 = balanced_accuracy_score(y_test, pred2)

  return bacc0, bacc1, bacc2

def getBacc():
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
  treeBacc = []
  forestBacc = []
  adaBacc = []
  for f in files:
    tBacc = []
    rBacc = []
    aBacc = []
    for i in range(5):
      t, r, a = bacc(f, i)
      tBacc.append(t)
      rBacc.append(r)
      aBacc.append(a)
    treeBacc.append(np.average(tBacc))
    forestBacc.append(np.average(rBacc))
    adaBacc.append(np.average(aBacc))
  return [treeBacc, forestBacc, adaBacc]

def test_ranked_acc():
  scores = np.vstack(getBacc()).T
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
  dataNames = ["TreeEnsemble", "RandomForest", "AdaBoost"]
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
  df.to_csv("rq2_bacc_table.csv", index=False)

  plt.show()
  return

def test_cd():
  # uses pairwise wilcoxon sign rank
  # has default value of 0.1 critical value for statistical test of difference
  treeEnsemble = [0.5654054854061151,
0.725,
0.07005023041650994,
0.9069656879024135,
0.3333333333333333,
0.3333333333333333,
0.5,
0.4924192909487027,
0.5413209282977085,
0.3215382205513785,
0.8357423205449521,
0.3333333333333333,
0.7655919021969376,
0.5,
0.9289711208618663,
0.582284977810599,
0.7085566381908416,
0.48755367501442637,
0.43100887406437394,
0.7147229330471355,
0.5,
0.3095238095238095]
  randomForest = [0.592860294052756,
      0.75,
      0.705802786964273,
      0.9907632238601962,
      0.49303830734537935,
      0.6074114409856026,
      0.5525474881499272,
      0.8162374499139204,
      0.7128257704627673,
      0.5502001413282878,
      0.872813845511214,
      0.9529708264381561,
      0.9908843516829353,
      0.7247781499888989,
      0.9774759792162696,
      0.878936720653881,
      0.9456892372892556,
      0.8820757708773259,
      0.9242418583026785,
      0.7143110249859252,
      0.9642339851233512,
      0.9206349206349206]
  adaBoost = [0.6414487504869781,
      0.7,
      0.0973725803650871,
      0.9467765154381503,
      0.46913922932143787,
      0.42888986126366824,
      0.5086184180636955,
      0.6658823529411764,
      0.6515989901910955,
      0.45160870679663584,
      0.9322953216374268,
      0.9375481940063551,
      0.7655919021969376,
      0.46199734608825516,
      0.9528147184503057,
      0.6248077220323836,
      0.7385688308383644,
      0.555142721184992,
      0.5753702526274453,
      0.6780040575234155,
      0.7448849802699745,
      0.6968253968253968]
  scores = np.vstack([treeEnsemble, randomForest, adaBoost]).T
  labels = ["TE", "RF", "AB"]
  files = ["balance-scale",
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
  fig, ax, p_values = plot_critical_difference(scores, labels, return_p_values=True)
  fig.savefig("cd_diag.png")
