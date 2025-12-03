import numpy as np
import pandas as pd
import pytest
import numpy as np
import pytest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from solution.DecisionStumpClassifier import DecisionStumpClassifier
from provided_code import data_loaders as dl

def getData(file_path):
  # load data for .arff files
  print(file_path)
  X,y = dl.load_tabular_xy(file_path)
  print(X.shape)
  print(X.dtype)
  print(X.ndim)
  return

#TODO: fertility should return float point error but doesnt...
def test_using_data():
  files = ["data/balance-scale/balance-scale.data",
           "data/balloons/balloons.data",
           "data/chess-krvk/chess-krvk.data",
           "data/chess-krvkp/chess-krvkp.data",
           "data/connect-4/connect-4.data",
           "data/contraceptive-method/contraceptive-method.data",
           "data/fertility/fertility.data",
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
  fileNames= ["balance-scale",
           "balloons",
           "chess-krvk",
           "chess-krvkp",
           "connect-4",
           "contraceptive-method",
           "fertility",
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

  datasets = []
  for i in range(23):
    X, y = dl.load_tabular_xy(files[i])
    datasets.append((X,y))

  summary_list = []

  for i, (X,y) in enumerate(datasets, start=1):
    n_samples = X.shape[0]
    n_features = X.shape[1]
    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)

    # Count unique categories per feature
    feature_cardinalities = [len(np.unique(X[:, idx])) for idx in
                             range(n_features)]

    # Build frequency map: {category_count: num_features}
    from collections import Counter
    freq = Counter(feature_cardinalities)

    # Format: "2→1, 5→2"
    feature_category_counts_str = ", ".join(
        f"{cat}:{count}" for cat, count in sorted(freq.items()))

    classes_str = ", ".join([str(c) for c in classes])
    class_dist_str = ", ".join([f"{cls}: {cnt}"
                                for cls, cnt in zip(classes, counts)])

    summary = {
      'Name': fileNames[i-1],
      'Samples': n_samples,
      'Features': n_features,
      '(category-count:feature-count)': feature_category_counts_str,
      'Number of classes': n_classes,
      'Class distribution': class_dist_str,
    }
    summary_list.append(summary)
    # Before creating the DataFrame
    for summary in summary_list:
      summary['Class distribution'] = str(summary['Class distribution'])

  df_summary = pd.DataFrame(summary_list)
  pd.set_option('display.max_colwidth', None)
  print(df_summary)
  df_summary.to_csv('dataset_summary.csv', index=False)

  assert True



