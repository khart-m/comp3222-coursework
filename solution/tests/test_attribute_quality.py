import pytest
from solution.part2 import (
    information_gain,
    information_gain_ratio,
    chi_squared,
    chi_squared_yates
)

# Testing. Create tests/test attribute quality.py that verifies each function matches
# the hand-worked results from Part 1 (allowing a small floating-point tolerance). You may as-
# sume the table input is valid; we do not require full input validation (e.g. checking non-negative
# counts). For the demo, also print each measure in the form:
# “<measure> for <attribute> = <value>”.
# Include a small if name == ’ main ’: block that runs these prints

def test_information_gain():
  inputP = [[4,0],[1,5]]
  expectedP = 0.60998
  actualP = information_gain(inputP)
  inputW = [[2,3],[3,2]]
  expectedW = 0.02904
  actualW = information_gain(inputW)
  inputS = [[2,5],[3,0]]
  expectedS = 0.39581
  actualS = information_gain(inputS)

  print("")
  assert actualP == pytest.approx(expectedP,0.001)
  print("information gain for peaty =", actualP)

  assert actualW == pytest.approx(expectedW,0.001)
  print("information gain for woody =", actualW)

  assert actualS == pytest.approx(expectedS,0.001)
  print("information gain for sweet = ", actualS)

def test_information_gain_ratio():
  inputP = [[4,0],[1,5]]
  expectedP = 0.6282
  actualP = information_gain_ratio(inputP)
  inputW = [[2,3],[3,2]]
  expectedW = 0.0290
  actualW = information_gain_ratio(inputW)
  inputS = [[2,5],[3,0]]
  expectedS = 0.4491
  actualS = information_gain_ratio(inputS)
  print("")

def test_chi_squared():
  inputP = [[4,0],[1,5]]
  expectedP = 20/3
  actualP = chi_squared(inputP)
  inputW = [[2,3],[3,2]]
  expectedW = 4/10
  actualW = chi_squared(inputW)
  inputS = [[2,5],[3,0]]
  expectedS = 30/7
  actualS = chi_squared(inputS)
  print("")
  assert actualP == pytest.approx(expectedP)
  print("chi-squared =", actualP)
  assert actualW == pytest.approx(expectedW)
  print("chi-squared =", actualW)
  assert actualS == pytest.approx(expectedS)
  print("chi-squared =", actualS)

if __name__ == '__main__':
  test_information_gain()
  test_chi_squared()