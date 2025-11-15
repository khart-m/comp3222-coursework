import unittest
import math

# -------------------------------------------------------
# IMPORT YOUR IMPLEMENTATION HERE
# -------------------------------------------------------
from part2 import (
    information_gain,
    information_gain_ratio,
    chi_squared,
    chi_squared_yates
)
# -------------------------------------------------------


class TestMetrics(unittest.TestCase):

    # Helper to print + assert
    def check_and_print(self, name, table, expected, actual):
        print(f"\n{name}")
        print(f"Table: {table}")
        print(f"Expected: {expected}")
        print(f"Actual:   {actual}")
        self.assertAlmostEqual(actual, expected, places=4)

    # ===============================================================
    # ORIGINAL INFORMATION GAIN TESTS
    # ===============================================================

    def test_information_gain_trivial(self):
        table = [[2, 2]]
        expected = 0.0
        actual = information_gain(table)
        self.check_and_print("Information Gain (Trivial)", table, expected, actual)

    def test_information_gain_symmetric(self):
        table = [[3, 1], [1, 3]]
        IG = 1 - (
            (4/8) * (-3/4*math.log2(3/4) - 1/4*math.log2(1/4)) * 2
        )
        expected = IG
        actual = information_gain(table)
        self.check_and_print("Information Gain (Symmetric)", table, expected, actual)

    def test_information_gain_perfect_split(self):
        table = [[5, 0], [0, 5]]
        expected = 1.0
        actual = information_gain(table)
        self.check_and_print("Information Gain (Perfect Split)", table, expected, actual)

    # ===============================================================
    # ORIGINAL GAIN RATIO TESTS
    # ===============================================================

    def test_gain_ratio_trivial(self):
        table = [[2, 2]]
        expected = 0.0
        actual = information_gain_ratio(table)
        self.check_and_print("Gain Ratio (Trivial)", table, expected, actual)

    def test_gain_ratio_symmetric(self):
        table = [[3, 1], [1, 3]]
        IG = 1 - (
            (4/8) * (-3/4*math.log2(3/4) - 1/4*math.log2(1/4)) * 2
        )
        expected = IG / 1
        actual = information_gain_ratio(table)
        self.check_and_print("Gain Ratio (Symmetric)", table, expected, actual)

    def test_gain_ratio_perfect_split(self):
        table = [[5, 0], [0, 5]]
        expected = 1.0
        actual = information_gain_ratio(table)
        self.check_and_print("Gain Ratio (Perfect Split)", table, expected, actual)

    # ===============================================================
    # ORIGINAL CHI-SQUARED TESTS
    # ===============================================================

    def test_chi_squared_trivial(self):
        table = [[2, 2]]
        expected = 0.0
        actual = chi_squared(table)
        self.check_and_print("Chi-Squared (Trivial)", table, expected, actual)

    def test_chi_squared_symmetric(self):
        table = [[3, 1], [1, 3]]
        expected = 2.0
        actual = chi_squared(table)
        self.check_and_print("Chi-Squared (Symmetric)", table, expected, actual)

    def test_chi_squared_perfect_split(self):
        table = [[5, 0], [0, 5]]
        expected = 10.0
        actual = chi_squared(table)
        self.check_and_print("Chi-Squared (Perfect Split)", table, expected, actual)

    # ===============================================================
    # ORIGINAL CHI-SQUARED YATES TESTS
    # ===============================================================

    def test_chi_squared_yates_trivial(self):
        table = [[2, 2]]
        expected = 0.0
        actual = chi_squared_yates(table)
        self.check_and_print("Chi-Squared Yates (Trivial)", table, expected, actual)

    def test_chi_squared_yates_symmetric(self):
        table = [[3, 1], [1, 3]]
        expected = 0.5
        actual = chi_squared_yates(table)
        self.check_and_print("Chi-Squared Yates (Symmetric)", table, expected, actual)

    def test_chi_squared_yates_perfect_split(self):
        table = [[5, 0], [0, 5]]
        expected = 6.4
        actual = chi_squared_yates(table)
        self.check_and_print("Chi-Squared Yates (Perfect Split)", table, expected, actual)

    # ===============================================================
    # NEW MULTI-ROW MULTI-CLASS TESTS (INFORMATION GAIN)
    # ===============================================================

    def test_information_gain_multirow_multiclass(self):
        table = [
            [10, 5, 5],
            [2, 8, 10],
            [4, 4, 12]
        ]
        expected = 1.5567 - 1.4560
        actual = information_gain(table)
        self.check_and_print("Information Gain (3x3)", table, expected, actual)

    def test_information_gain_skewed_distribution(self):
        table = [
            [40, 5, 5],
            [2, 3, 5],
            [1, 1, 3]
        ]
        expected = 0.2842
        actual = information_gain(table)
        self.check_and_print("Information Gain (Skewed 3x3)", table, expected, actual)

    def test_information_gain_large_counts(self):
        table = [
            [100, 150, 250],
            [40, 60, 100],
            [10, 20, 30],
            [5, 10, 15]
        ]
        expected = 0.0066
        actual = information_gain(table)
        self.check_and_print("Information Gain (Large 4x3)", table, expected, actual)

    # ===============================================================
    # NEW MULTI-ROW MULTI-CLASS TESTS (GAIN RATIO)
    # ===============================================================

    def test_gain_ratio_multirow_multiclass(self):
        table = [
            [10, 5, 5],
            [2, 8, 10],
            [4, 4, 12]
        ]
        expected = 0.0738
        actual = information_gain_ratio(table)
        self.check_and_print("Gain Ratio (3x3)", table, expected, actual)

    def test_gain_ratio_balanced_rows(self):
        table = [
            [5, 5, 5],
            [5, 5, 5],
            [5, 5, 5]
        ]
        expected = 0.0
        actual = information_gain_ratio(table)
        self.check_and_print("Gain Ratio (Balanced 3x3)", table, expected, actual)

    def test_gain_ratio_multiple_rows_unequal(self):
        table = [
            [20, 10, 5],
            [3, 6, 9],
            [4, 8, 4],
            [1, 2, 7]
        ]
        expected = 0.1024
        actual = information_gain_ratio(table)
        self.check_and_print("Gain Ratio (4x3 Unequal)", table, expected, actual)

    # ===============================================================
    # NEW MULTI-ROW MULTI-CLASS TESTS (CHI-SQUARED)
    # ===============================================================

    def test_chi_squared_multirow_multiclass(self):
        table = [
            [10, 5, 5],
            [2, 8, 10],
            [4, 4, 12]
        ]
        expected = 7.508
        actual = chi_squared(table)
        self.check_and_print("Chi-Squared (3x3)", table, expected, actual)

    def test_chi_squared_balanced(self):
        table = [
            [5, 5, 5],
            [5, 5, 5],
            [5, 5, 5]
        ]
        expected = 0.0
        actual = chi_squared(table)
        self.check_and_print("Chi-Squared (Balanced 3x3)", table, expected, actual)

    def test_chi_squared_large(self):
        table = [
            [200, 150, 50],
            [40, 70, 90],
            [10, 20, 30]
        ]
        expected = 26.655
        actual = chi_squared(table)
        self.check_and_print("Chi-Squared (Large 3x3)", table, expected, actual)

    # ===============================================================
    # NEW MULTI-ROW + YATES TESTS
    # ===============================================================

    def test_chi_squared_yates_multirow(self):
        table = [
            [10, 5, 5],
            [2, 8, 10],
            [4, 4, 12]
        ]
        expected = chi_squared(table)
        actual = chi_squared_yates(table)
        self.check_and_print("Chi-Squared Yates (3x3 → normal)", table, expected, actual)

    def test_chi_squared_yates_unbalanced_2x2(self):
        table = [
            [30, 5],
            [10, 20]
        ]
        expected = 5.186
        actual = chi_squared_yates(table)
        self.check_and_print("Chi-Squared Yates (Unbalanced 2x2)", table, expected, actual)

    def test_chi_squared_yates_high_counts(self):
        table = [
            [100, 40],
            [30, 80]
        ]
        expected = 24.384
        actual = chi_squared_yates(table)
        self.check_and_print("Chi-Squared Yates (High Count 2x2)", table, expected, actual)


if __name__ == "__main__":
    unittest.main()
