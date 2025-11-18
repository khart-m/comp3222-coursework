from enum import unique

import numpy as np


#Input format. Assume rows correspond to distinct values of the candidate attribute and columns
#correspond to class labels. For example, for Peaty (yes/no) versus Region (Islay/Speyside) us-
#ing Table 1, the contingency table is

#i,s
#4,0 peaty
#1,5 not peaty

#on peaty

#since there are 4 peaty Islay whiskies, 0 peaty Speyside whiskies, 1 non-peaty Islay whisky, and
#5 non-peaty Speyside whiskies


def information_gain(table):
  numberOfAttributes = len(table)
  numberOfClasses = len(table[0])

  # work out parent node entropy
  parentEntropy = 0
  totalCount = 0
  for i in range(numberOfAttributes):
    totalCount += sum(table[i])
  #print("totalCount = ", totalCount)
  for j in range(numberOfClasses):
    attIcount=0
    for i in range(numberOfAttributes):
      #print(table[i][j])
      attIcount += table[i][j]
    parentEntropy += (attIcount / totalCount) * (np.log2(attIcount / totalCount))

  parentEntropy = - parentEntropy
  #print("parent entropy: ", parentEntropy)

  # work out weighted entropy at each node
  nodeEntropy = 0
  for i in range(numberOfAttributes):
    total = sum(table[i]) # this is the total of things which are this attribute (e.g. peaty)
    for j in range(numberOfClasses):
      if(table[i][j]/total != 0):
        nodeEntropy += total/totalCount * (table[i][j] / total) * (np.log2(table[i][j]/total))
        #print(nodeEntropy)

    #print("node entropy: ", nodeEntropy)

  nodeEntropy = - nodeEntropy
  gain = parentEntropy - nodeEntropy
  return gain

def split_information(table):
  # row over total = x
  # sum (x * log(x))
  numberOfAttributes = len(table)
  sInfo = 0
  totalCount = 0
  x = 0
  for i in range(numberOfAttributes):
    totalCount += sum(table[i])
    print(totalCount)
  for i in range(numberOfAttributes):
    x = sum(table[i])/totalCount
    if(x != 0):
      sInfo += (x * np.log2(x))
  return sInfo

def information_gain_ratio(table):
  ratio = 0
  if(split_information(table) != 0):
    ratio = information_gain(table)/split_information(table)
  return - ratio

def chi_squared(table):
  numberOfAttributes = len(table)
  numberOfClasses = len(table[0])
  chi2 = 0
  totalCount = 0
  for i in range(numberOfAttributes):
    totalCount += sum(table[i])

  for i in range(numberOfAttributes):
    attSum = sum(table[i])

    for j in range(numberOfClasses):
      classSum = 0
      for k in range(numberOfAttributes):
        classSum += table[k][j]
      expected = (attSum * classSum) / totalCount
      actual = table[i][j]
      chi2 += ((actual - expected )**2)/expected

  return chi2

def chi_squared_yates(table):
  numberOfAttributes = len(table)
  numberOfClasses = len(table[0])
  if(numberOfAttributes==2 & numberOfClasses==2):
    chi2 = 0
    totalCount = 0

    for i in range(numberOfAttributes):
      totalCount += sum(table[i])

    for i in range(numberOfAttributes):
      attSum = sum(table[i])

      for j in range(numberOfClasses):
        classSum = 0
        for k in range(numberOfAttributes):
          classSum += table[k][j]
        expected = (attSum * classSum) / totalCount
        actual = table[i][j]
        chi2 += ((abs(actual - expected)-0.5) ** 2) / expected
    return chi2
  else:
    return chi_squared(table)


def build_table(attr_col, class_labels, n_classes):
  unique_vals = np.unique(attr_col)
  n_vals = len(unique_vals)
  table = np.zeros((n_vals, n_classes), dtype=int)
  value_to_row = {val: i for i, val in enumerate(unique_vals)}
  for attr_val, class_label in zip(attr_col, class_labels):
    row = value_to_row[attr_val]
    col = int(class_label)
    table[row, col] += 1
  return table


def measure_quality(measure, table):
  if measure == "ig":
    return information_gain(table)

  if measure == "gain_ratio":
    return information_gain_ratio(table)

  if measure == "chi2":
    return chi_squared(table)

  if measure == "chi2_yates":
    return chi_squared_yates(table)
  return 0

def fit(X, y):

  classes = np.unique(y)
  n_classes_ = classes.shape[0]

  best_score = 0
  best_att = None
  best_index = None
  best_table = None

  transposed_X = X.T

  index = 0
  for att in transposed_X:
    table = build_table(att, y, n_classes_)
    quality = measure_quality("ig", table)
    if (best_att is None) or (quality > best_score):
      best_score = quality
      best_att = att
      best_index = index
      best_table = table
    index += 1

  print("Best attribute ", best_att)
  print("Best index ", best_index)
  print("Best table ", best_table)
  print("Best quality ", best_score)
  pass

peaty = np.array([1,1,1,1,0,0,0,0,0,0])
woody  = np.array(["no","yes","no","no","yes","yes","yes","yes","no","no"])
sweet  = np.array(["yes","yes","no","no","no","yes","yes","yes","yes","yes"])
region = np.array([0,0,0,0,0,1,1,1,1,1])


class_labels = region

n_classes = 2

# Example: build table for "Peaty"
table_peaty = build_table(peaty, class_labels, n_classes)
print(table_peaty)

# # Example: build table for "Woody"
# table_woody = build_table(woody, class_labels, n_classes)
# print(table_woody)
#
# # Example: build table for "Sweet"
# table_sweet = build_table(sweet, class_labels, n_classes)
# print(table_sweet)

X = np.array([[1,1,1,1,0,0,0,0,0,0],["no","yes","no","no","yes","yes","yes","yes","no","no"], ["yes","yes","no","no","no","yes","yes","yes","yes","yes"]])
y = region
fit(X.T, y)
testData = [[4,0],[1,5]]

print("array length: ", len(X.T))
#0.6283
#print(information_gain_ratio(testData))
#print("chisquared: " , chi_squared(testData))
#print(chi_squared_yates(testData))