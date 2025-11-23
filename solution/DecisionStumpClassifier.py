from numba.core.ir_utils import find_max_label
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
import numpy as np
import pandas as pd
from solution.part2 import (
    information_gain,
    information_gain_ratio,
    chi_squared,
    chi_squared_yates
)

class DecisionStumpClassifier(BaseEstimator, ClassifierMixin):
    """Add you docstring here.
    """

    def __init__(self,
                n_attributes: int | None = None,
                quality_measure: str = "ig",
                random_state: int | None = None,
                alpha: float = 1.0):
      # n_attributes - number of attributes
      # quality_measure - where ig = information gain
      # random_state - used for random
      # alpha - ?
      self.n_attributes = n_attributes
      self.quality_measure = quality_measure
      self.random_state = random_state
      self.alpha = alpha

      # Attributes to be set in fit
      self.att_index: int | None = None
      self.att_table: [[int]] | None = None
      self.att_categories: dict[str,int] | None = None

      self.root_probs: [int] | None = None

      self.feature_values_ = None
      self.class_counts_ = None  # mapping value -> counts array of shape (n_classes,)
      self.classes_ = None
      self.n_classes_ = None

    def check_data(self, X):
      """
      Ensure X contains only categorical (string or integer) features.
      Parameters
      ----------
      X - the data to check

      Returns
      -------
      - raises error if data is wrong
      """
      # Case: pandas DataFrame
      if isinstance(X, pd.DataFrame):
        for col, dtype in X.dtypes.items():
          if pd.api.types.is_float_dtype(dtype):
            raise TypeError(
                f"Column '{col}' is float-valued; only categorical (int or string) "
                f"features are allowed."
            )
        return

      # Case: numpy array
      if isinstance(X, np.ndarray):
        if np.issubdtype(X.dtype, np.floating):
          raise TypeError(
              "X contains float values; only categorical (int or string) "
              "features are allowed."
          )
        # If dtype=object, ensure no floats hidden inside
        if X.dtype == object:
          if any(isinstance(v, float) for v in X.ravel()):
            raise TypeError(
                "X contains float values inside an object array; "
                "only categorical features are allowed."
            )
        return
      # Otherwise unsupported type
      raise TypeError("X must be a numpy array or a pandas DataFrame.")

    def build_table(self, attr_col, class_labels, n_classes):
      """
      When given categorical attributes and classes in any form, creates a contingency table
      Parameters
      ----------
      attr_col - the attribute columns
      class_labels - the class labels
      n_classes - the number of classes

      Returns
      -------
      - contingency table
      """
      col = self.clean_column(attr_col)
      unique_vals = np.unique(col)
      n_vals = len(unique_vals)
      table = np.zeros((n_vals, n_classes), dtype=int)
      value_to_row = {val: i for i, val in enumerate(unique_vals)}
      self.att_categories = value_to_row
      for attr_val, class_label in zip(col, class_labels):
        row = value_to_row[attr_val]
        col = int(class_label)
        table[row, col] += 1
      return table

    def clean_column(self, col):
      """
      Helper function to deal with missing values
      Parameters
      ----------
      col - column

      Returns
      -------
      column where missing values are replaced with a placeholder
      """
      cleaned = []
      for v in col:
        if v is None:
          cleaned.append("<<MISSING>>")
        elif isinstance(v, float) and np.isnan(v):
          cleaned.append("<<MISSING>>")
        elif v == "":
          cleaned.append("<<MISSING>>")
        else:
          cleaned.append(v)
      return np.array(cleaned, dtype=object)

    def quality(self, table):
      """
      measures the quality given the quality measure defined in self using the helper functions from part1
      Parameters
      ----------
      table - contingency table

      Returns
      -------
      - float describing the quality of a split
      """
      if self.quality_measure == "ig":
        return information_gain(table)

      if self.quality_measure == "gain_ratio":
        return information_gain_ratio(table)

      if self.quality_measure == "chi2":
        return chi_squared(table)

      if self.quality_measure == "chi2_yates":
        return chi_squared_yates(table)
      return 0

    def fit(self,X, y):
      """
      Train the classifier, by picking an attribute to split on
      Parameters
      ----------
      X - train data
      y - class labels

      Returns
      -------
      - nothing, but stores information about the training in self.att_index, self.att_table, self.root_probs
      """
      # X should be [[att1], [att2], ...]
      # y should be [class, class, ...]
      self.check_data(X)

      """
      # check and format y
      y_arr = np.asarray(y)
      if(y_arr.ndim != 1):
        raise ValueError("y not 1D array")
      if not np.issubdtype(y_arr.dtype, np.floating):
        try:
          y_arr = y_arr.astype(int)
        except Exception:
          raise TypeError("y must be integer-encoded class labels 0..c-1.")

      # check and format x
      if isinstance(X, pd.DataFrame):
        X_df = X.reset_index(drop=True)
        n_samples, n_features = X_df.shape
      else:
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
          # single feature column
          X_arr = X_arr.reshape(-1, 1)
        n_samples, n_features = X_arr.shape
        # for uniform handling below, wrap numpy array as X_arr
        X_df = None

        if n_samples != y_arr.shape[0]:
          raise ValueError("Number of samples in X and y do not match.")

      """
      # classes
      classes = np.unique(y)
      self.classes_ = classes
      self.n_classes_ = classes.shape[0]

      # pick best attribute to split on
      best_score = 0
      best_att = None
      best_index = None
      best_table = None

      transposed_X = X.T

      # random sampling
      n_features = X.shape[1]

      if self.n_attributes is None:
        candidate_indices = range(n_features)
      else:
        rng = np.random.default_rng(self.random_state)
        candidate_indices = rng.choice(
            n_features,
            size=self.n_attributes,
            replace=False
        )

      index = 0
      for index in candidate_indices:
        att = transposed_X[index]
        table = self.build_table(att, y, self.n_classes_)
        quality = self.quality(table)
        if (best_att is None) or (quality > best_score):
          best_score = quality
          best_att = att
          best_index = index
          best_table = table
        index += 1

      self.att_index = best_index
      self.att_table = best_table
      print("best_att", best_att)
      print("self.att_index", self.att_index)
      print("self.att_table", self.att_table)

      totalCases = len(y)
      root_probs = np.zeros(self.n_classes_, dtype=float)
      print("setting the root table... ")
      for i, cls in enumerate(classes):
        count = np.sum(y == cls)
        root_probs[i] = (count + self.alpha) / (totalCases + self.alpha * self.n_classes_)
      self.root_probs = root_probs


      pass

    def predict(self, X):
        """
        uses predict_proba to predict class values for records given some attribute values x
        Parameters
        ----------
        X - input data

        Returns
        -------
        - predicted class values
        """
        output = np.zeros(len(X), dtype=int)
        probs = self.predict_proba(X)
        i = 0
        for prob in probs:
          output[i] = self.find_max_label(prob)
          i += 1
        return output

    def find_max_label(self, prob):
      """
      helper function to find the most probable label for a given probability, tie breaks using lowest index
      Parameters
      ----------
      prob - list of probabilities for each class value

      Returns
      -------
      - the max index of the highest probability
      """
      max = 0
      max_index = 0
      for i in range(len(prob)):
        if prob[i] > max:
          max = prob[i]
          max_index = i

      return max_index

    def predict_proba(self, X):
        """
        predicts the probabilities of each class value for given data X
        Parameters
        ----------
        X - input data

        Returns
        -------
        2d array of probabilities
        """
        # where x is [[1,0,1,1],[2,... and each inside array is a case i.e. X[1] predicts output[1]
        # should output smth like the probability of it being each class for each input based on the table + alpha e.g. if there are three options, class1 class2 class3:
        # [[0.3,0.3,0.4],....[p1,p2,p3]]

        # initialise arrays
        output = np.zeros((len(X), self.n_classes_), dtype=float)

        n_classes = self.n_classes_

        # for each row in x, what is attribute at the index
        # based on that find the probability of it being each class
        # p(class|value) = (count(v,c) + a)/(sumj(count(v,j) + a))
        # from the att_table we can see [[1,5,2],[4,0,1]] -
        # if the case has the attribute, it has 1/8 chance of class 0, 5/8 of class 1, 2/8 of class 3
        # if it doesn't have the attribute, it has 4/5 chance of class 0, 0/5 of class 2, 1/5 of class 3
        value_to_row = self.att_categories
        print("value to rows", value_to_row)
        c = 0
        for case in X:
          print("case", case)
          v = case[self.att_index]
          if v is None or (isinstance(v, float) and np.isnan(v)) or v == "":
            v = "<<MISSING>>"

          if v in value_to_row:
            row = value_to_row[v]
            counts = self.att_table[row]
            total = sum(counts) + (self.alpha * n_classes)

            for classV in range(n_classes):
              count = counts[classV] + self.alpha
              p = (count) / total
              print("inserting ", p, "into ", c, ",", classV)
              output[c, classV] = p
              print("output[c,classV]", output)
              print(output[c, classV])
          else:
            #unseen categories at prediction - use root priors probs
            probs = self.root_probs
            output[c] = probs




          c += 1

        print("output: ", output)
        return output