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
      print(table[i][j])
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
        print(nodeEntropy)

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



testData = [[4,0],[1,5]]
#0.6283
print(information_gain_ratio(testData))
print("chisquared: " , chi_squared(testData))
print(chi_squared_yates(testData))