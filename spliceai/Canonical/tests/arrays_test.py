import unittest

import numpy as np

from modular_splicing.utils.arrays import add_cl_in


class TestAddCLIn(unittest.TestCase):
    def add_partial_cl_in_4_test(self):
        actual = add_cl_in(
            np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]), cl=4
        )
        expected = [
            [0, 0, 1, 2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7, 8, 9, 10],
            [7, 8, 9, 10, 11, 12, 0, 0],
        ]
        self.assertEqual(actual.tolist(), expected)

    def add_partial_cl_in_2_test_different_padding(self):
        actual = add_cl_in(
            np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]), cl=4, pad_value=-1
        )
        expected = [
            [-1, -1, 1, 2, 3, 4, 5, 6],
            [3, 4, 5, 6, 7, 8, 9, 10],
            [7, 8, 9, 10, 11, 12, -1, -1],
        ]
        self.assertEqual(actual.tolist(), expected)

    def add_partial_cl_in_2_test(self):
        actual = add_cl_in(
            np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]), cl=2
        )
        expected = [
            [0, 1, 2, 3, 4, 5],
            [4, 5, 6, 7, 8, 9],
            [8, 9, 10, 11, 12, 0],
        ]
        self.assertEqual(actual.tolist(), expected)

    def no_cl_test(self):
        actual = add_cl_in(
            np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]), cl=0
        )
        expected = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ]
        self.assertEqual(actual.tolist(), expected)

    def full_cl_test(self):
        actual = add_cl_in(
            np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]), cl=8
        )
        expected = [
            [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            [5, 6, 7, 8, 9, 10, 11, 12, 0, 0, 0, 0],
        ]
        self.assertEqual(actual.tolist(), expected)
        self.assertEqual(actual.tolist(), expected)
