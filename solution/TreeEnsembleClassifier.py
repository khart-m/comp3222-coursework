import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from DecisionStumpClassifier import DecisionStumpClassifier


class TreeEnsembleClassifier(BaseEstimator, ClassifierMixin):
  """
  Parameters
  ----------
  base_estimator : estimator
      Any sklearn-compatible classifier (must implement fit/predict; predict_proba optional).
  n_estimators : int, default=10
      Number of cloned members.
  sample_fraction : float, default=0.6
      Fraction of training samples used per member (rounded to at least 1).
  replace : bool, default=False
      If True, sample with replacement (bootstrap). Default is without replacement.
  random_state : int or None, default=None
      Seed for reproducibility.
  """

  def __init__(self, base_estimator=None, n_estimators=100, average_probas=True, random_state=None, ):
        self.base_estimator = base_estimator
        if base_estimator is None:
          self.base_estimator = DecisionStumpClassifier()
        self.n_estimators = n_estimators # ensemble size
        self.average_probas = average_probas # majority vote (hard voting) and average predicted probabilities (soft voting)
        self.random_state = random_state
        self.features_subsets_ = []
        self.quality_measures_ = []
        self.estimators_ = []
        self.classes_ = 0
  def validate_inputs(self, X):
    """
    This ensemble should support categorical inputs only
    If any real-valued float attributes are detected in X, raise a ValueError (consistent with the stump)
    Parameters
    ----------
    X - training data
    Returns
    -------
    nothing - raise a ValueError if float attributes detected
    """
  pass

  def random_sample(self, X, y):
    """
    helper function to randomly select a subset of the training data with replacement
    Parameters
    ----------
    X - training data
    y - training labels

    Returns
    -------
    2dx and 2dy with
    """
    X = np.asanyarray(X)
    y = np.asanyarray(y)

    n_samples = len(X)
    rng = np.random.default_rng(self.random_state)

    sampled_X = []
    sampled_y = []

    for _ in range(self.n_estimators):
      idx = rng.integers(0, n_samples, size=n_samples)
      sampled_X.append(X[idx])
      sampled_y.append(y[idx])

    return sampled_X, sampled_y

  def random_features(self, sampled_X):
    """
    apply feature subspace selection to each random sample
    Parameters
    ----------
    sampled_X - sampled data

    Returns
    -------
    sampled data with random features chosen for each one
    """
    rng = np.random.default_rng(self.random_state)
    random_features_X = []
    for X_i in sampled_X:
      n_features_total = X_i.shape[1]

      # random forest rule: m = sqrt(p)
      m = max(1, int(np.sqrt(n_features_total)))

      feat_idx = rng.choice(n_features_total, size=m, replace=False)
      self.features_subsets_.append(feat_idx)
      random_features_X.append(X_i[:, feat_idx])
    return random_features_X

  def random_quality(self):
    """
    generates a list of random quality measures the length of the number of estimators
    Returns
    -------
    list of quality measures
    """
    rng = np.random.default_rng(self.random_state)
    measures = ["ig", "gain_ratio", "chi2", "chi2_yates"]

    self.quality_measures_ = rng.choice(measures, size=self.n_estimators, replace=True)

    return self.quality_measures_

  def boost(self):
    pass

  def fit(self, X, y):
    """
    Validate inputs
    Create and store an array estimators_ of base classifiers
      Construct each stump according to your diversity strategy and fit independently
      Store classes_ - from np.unique(y) in a stable order
        Map y onto 0,c-1 before fitting base classifiers?
          Using scikit LabelEncoder
    Diversity strategies:
    - data sample manipulation - take different samples of training data for each classifier
    - input feature manipulation - train models on different features
    - learning parameter manipulation - train models w/ different quality measures
    Parameters
    ----------
    X
    y

    Returns
    -------

    """
    self.validate_inputs(X)
    sampled_X, sampled_y = self.random_sample(X, y)
    reduced_X = self.random_features(sampled_X)
    random_qualities = self.random_quality()

    self.classes_ = np.unique(y)

    for i in range(self.n_estimators):
      est = clone(self.base_estimator)
      est.quality_measure = random_qualities[i]

      Xi = reduced_X[i]
      yi = sampled_y[i]

      est.fit(Xi, yi)
      self.estimators_.append(est)

    return self

  def predict(self, X):

    # structure to hold the predictions

    preds = np.empty((self.n_estimators, len(X)), dtype=object)

    for i, (est, feat_idx) in enumerate(zip(self.estimators_, self.features_subsets_)):
      Xi = X[:, feat_idx]
      preds[i] = est.predict(Xi)

    if not self.average_probas:
      print("returning majority class")
      final_preds = np.empty(len(X), dtype=object)
      for i in range(len(X)):
        votes = preds[:, i]
        classes, counts = np.unique(votes, return_counts=True)

        final_pred = classes[np.argmax(counts)]
        final_preds[i] = final_pred

      return final_preds
    else:
      print("calling predict proba - and returning argmax")
      probas = self.predict_proba(X)
      class_indices = np.argmax(probas, axis=1)
      return self.classes_[class_indices]


  def predict_proba(self, X):

    if self.average_probas:
      # will hold probabilities for each class, for each row of data
      sum_probas = np.zeros((len(X), len(self.classes_)), dtype=float)

      # loop through all estimators
      for est, feat_idx in zip(self.estimators_, self.features_subsets_):
        Xi = X[:, feat_idx]

        est_proba = est.predict_proba(Xi)
        est_classes = est.classes_

        # because estimators might have different class orders
        aligned = np.zeros((len(X), len(self.classes_)), dtype=float)

        for i, cls in enumerate(est_classes):
          col_index = np.where(self.classes_ == cls)[0][0]
          aligned[:, col_index] = est_proba[:, i]

        sum_probas += aligned

      prob_avg = sum_probas/self.n_estimators
      return prob_avg

    else:

      vote_counts = np.zeros((len(X), len(self.classes_)), dtype=int)
      for est, feat_idx in zip(self.estimators_, self.features_subsets_):
        Xi = X[:, feat_idx]
        preds = est.predict(Xi)
        for i in range(len(X)):
          voted_class = preds[i]
          cls_index = np.where(self.classes_ == voted_class)[0][0]
          vote_counts[i, cls_index] += 1
      vote_proportions = vote_counts/self.n_estimators
    return vote_proportions

