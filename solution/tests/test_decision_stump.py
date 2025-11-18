import numpy as np
import pytest
import numpy as np
import pytest

from solution.DecisionStumpClassifier import DecisionStumpClassifier


# -------------------------------------------------------------------------
# Helper: whisky dataset
# -------------------------------------------------------------------------
def whisky_data():
    X = [
        ["yes", "no",  "yes"],
        ["yes", "yes", "yes"],
        ["yes", "no",  "no"],
        ["yes", "no",  "no"],
        ["no",  "yes", "no"],
        ["no",  "yes", "yes"],
        ["no",  "yes", "yes"],
        ["no",  "yes", "yes"],
        ["no",  "no",  "yes"],
        ["no",  "no",  "yes"],
    ]

    # Encode labels as 0 = Islay, 1 = Speyside
    y = np.array([0,0,0,0,0, 1,1,1,1,1])
    return np.array(X, dtype=object), y


# -------------------------------------------------------------------------
# Test 1: floats in X must raise an exception
# -------------------------------------------------------------------------
def test_floats_raise_exception():
    X = np.array([
        ["a", "b"],
        ["a", 2.5],   # float present
    ], dtype=object)
    y = np.array([0, 1])

    stump = DecisionStumpClassifier()

    with pytest.raises((TypeError, ValueError)):
        stump.fit(X, y)


# -------------------------------------------------------------------------
# Test 2: deterministic attribute choice when random_state is set
# -------------------------------------------------------------------------
def test_deterministic_attribute_choice():
    X, y = whisky_data()

    # restrict to subset of attributes so sampling matters
    stump1 = DecisionStumpClassifier(n_attributes=2, random_state=123)
    stump2 = DecisionStumpClassifier(n_attributes=2, random_state=123)

    stump1.fit(X, y)
    stump2.fit(X, y)

    assert stump1.att_index == stump2.att_index, \
        "Attribute index should be deterministic when random_state is fixed"


# -------------------------------------------------------------------------
# Test 3: predict_proba returns valid probability vectors in class order
# -------------------------------------------------------------------------
def test_predict_proba_validity():
    X, y = whisky_data()
    stump = DecisionStumpClassifier()
    stump.fit(X, y)

    probs = stump.predict_proba(X)

    assert probs.shape == (len(X), len(np.unique(y)))

    # rows must sum to 1
    row_sums = probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0), "Each probability vector must sum to 1"

    # all probabilities must be >= 0
    assert np.all(probs >= 0.0)


# -------------------------------------------------------------------------
# Test 4: unseen category falls back to root prior
# -------------------------------------------------------------------------
def test_unseen_category_fallback():
    X, y = whisky_data()
    stump = DecisionStumpClassifier()
    stump.fit(X, y)

    # create unseen category case
    X_unseen = np.array([["maybe", "no", "yes", "UnknownRegion"]], dtype=object)

    probs = stump.predict_proba(X_unseen)[0]

    # root prior = overall class frequencies
    root_prior = np.bincount(y) / len(y)

    assert np.allclose(probs, root_prior), \
        "Unseen category should fall back to root prior"


# -------------------------------------------------------------------------
# Test 5: missing values treated as separate category
# -------------------------------------------------------------------------
def test_missing_values_treated_as_category():
    X, y = whisky_data()

    stump = DecisionStumpClassifier()
    stump.fit(X, y)

    # Predict with a missing value
    X_missing = np.array([[None, "no", "yes", "Islay"]], dtype=object)

    # Should not raise, should produce a valid probability vector
    probs = stump.predict_proba(X_missing)[0]

    assert np.isclose(probs.sum(), 1.0)
    assert np.all(probs >= 0.0)

def test_predict():
  X, y = whisky_data()
  stump = DecisionStumpClassifier()
  stump.fit(X, y)

  result = stump.predict(X)
  print(result)
  assert result.all() == y.all()