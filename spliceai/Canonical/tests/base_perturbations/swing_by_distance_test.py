import unittest
from parameterized import parameterized

import numpy as np

from modular_splicing.base_perturbations.swing_by_distance import (
    distance_to_feature_center,
    is_kaway,
)


class SwingByDistanceTest(unittest.TestCase):
    @parameterized.expand([(x,) for x in range(100)])
    def test_kaway_no_distance(self, seed):
        x = np.random.RandomState(seed).choice(2, size=(10, 1000))
        self.assertTrue((x == is_kaway(x, 0)).all())

    def test_both_directions(self):
        vals = np.array(
            [
                [[1, 0, 0, 0, 1, 0, 0, 0, 0, 0]],
                [[0, 1, 0, 1, 0, 1, 0, 0, 0, 0]],
                [[0, 0, 1, 0, 0, 0, 1, 0, 0, 0]],
                [[0, 1, 0, 1, 0, 0, 0, 1, 0, 0]],
            ],
        )
        for k in range(vals.shape[0]):
            self.assertTrue((vals[k] == is_kaway(vals[0], k)).all())

    def test_distance_to_motif_center(self):
        vals = np.array(
            [
                [[1, 0, 0, 0, 1, 0, 0, 0, 0, 0]],
                [[0, 1, 2, 1, 0, 1, 2, 3, 4, 5]],
                [[0, 1, 2, 1, 0, 1, 2, 3, 4, 4]],
            ],
        )
        self.assertTrue((vals[1] == distance_to_feature_center(vals[0], 100)).all())
        self.assertTrue((vals[2] == distance_to_feature_center(vals[0], 4)).all())
