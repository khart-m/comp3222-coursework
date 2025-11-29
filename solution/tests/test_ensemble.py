import numpy as np
import pytest
from TreeEnsembleClassifier import TreeEnsembleClassifier
from provided_code import data_loaders as dl


def load_data():
    X, y = dl.load_tabular_xy("data/balance-scale/balance-scale.data")
    return X, y


def test_fit_debug():
    X, y = load_data()

    clf = TreeEnsembleClassifier(
        n_estimators=5,
        average_probas=True,
        random_state=0
    )

    print("\n===== Calling fit() =====")
    clf.fit(X, y)

    print("\n===== INTERNAL STATE AFTER FIT =====")

    # Estimators
    print("Number of estimators:", len(clf.estimators_))

    # Classes
    print("Classes:", clf.classes_)

    # Feature subsets
    print("\nFeature subsets:")
    for i, fs in enumerate(clf.features_subsets_):
        print(f"  Estimator {i}: {fs}")

    # Quality measures
    print("\nQuality measures:")
    print(clf.quality_measures_)

    # Estimator types
    print("\nEstimator types:")
    for i, est in enumerate(clf.estimators_):
        print(f"  Estimator {i}: {type(est)}")

    print("\n===== END DEBUG OUTPUT =====\n")

    # Basic assertions so pytest considers the test "passed"
    assert len(clf.estimators_) == clf.n_estimators
    assert len(clf.features_subsets_) == clf.n_estimators
    assert len(clf.quality_measures_) == clf.n_estimators
    assert clf.classes_ is not None


def test_predict_debug():
  X, y = load_data()

  clf = TreeEnsembleClassifier(
      n_estimators=10,
      average_probas=False,
      random_state=0
  )

  # First, fit the classifier
  print("\n===== Calling fit() =====")
  clf.fit(X, y)

  print("\n===== Calling predict() =====")
  predictions = clf.predict(X)
  print("Predictions:", predictions)
  print("classes: ", clf.classes_)

  for est in clf.estimators_:
    print("estimator ", est.att_index_, est.quality_measure)

  # Check that predictions have the same number of samples as X
  assert len(predictions) == X.shape[0]

  # Check that all predicted labels are in the known classes
  for pred in predictions:
    assert pred in clf.classes_

  # Optionally, check soft voting probabilities if implemented
  if hasattr(clf, "predict_proba"):
    probas = clf.predict_proba(X)
    print("\nPredicted probabilities:")
    print(probas)
    # Probabilities shape check
    assert probas.shape == (X.shape[0], len(clf.classes_))
    # Probabilities sum to 1
    assert np.allclose(probas.sum(axis=1), 1.0)

  print("\n===== END PREDICT DEBUG OUTPUT =====\n")

def test_predict_functions():
    X, y = load_data()
    clf = TreeEnsembleClassifier(
        n_estimators=5,
        average_probas=False,
        random_state=42
    )

    clf.fit(X, y)

    print("\n===== Testing predict() =====")
    y_pred = clf.predict(X)
    print("Predictions:", y_pred)

    # Check length matches number of samples
    assert len(y_pred) == X.shape[0]

    # Check predictions are valid classes
    for pred in y_pred:
      assert pred in clf.classes_

    print("\n===== Testing predict_proba() =====")
    proba = clf.predict_proba(X)
    print("Predicted probabilities:\n", proba)

    # Check shape: n_samples x n_classes
    assert proba.shape == (X.shape[0], len(clf.classes_))

    # Check that each row sums to 1
    row_sums = proba.sum(axis=1)
    for s in row_sums:
      np.testing.assert_almost_equal(s, 1.0, decimal=6)

    # Check all probabilities are between 0 and 1
    assert np.all(proba >= 0) and np.all(proba <= 1)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

def test_ensemble_accuracy():
    X, y = load_data()

    # Split into train and test sets (e.g., 60% train, 40% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    clf = TreeEnsembleClassifier(
        n_estimators=5,
        average_probas=True,
        random_state=0
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # Calculate accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.2f}")

    # Assert accuracy is within a reasonable range (>= 0.5 for dummy data)
    assert 0.0 <= acc <= 1.0
    assert acc >= 0.5
