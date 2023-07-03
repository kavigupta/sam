import unittest

from modular_splicing.motif_perturbations.summarize_effect import avg_blur, radial_sum


class TestSum(unittest.TestCase):
    def test_sum_basic(self):
        self.assertEqual(radial_sum([1, 2, 3, 4, 5], 0).tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(radial_sum([1, 2, 3, 4, 5], 1).tolist(), [3, 6, 9, 12, 9])
        self.assertEqual(radial_sum([1, 2, 3, 4, 5], 2).tolist(), [6, 10, 15, 14, 12])

    def test_blur(self):
        self.assertEqual(avg_blur([1, 2, 3, 4, 5], 0).tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(avg_blur([1, 2, 3, 4, 5], 1).tolist(), [1.5, 2, 3, 4, 4.5])
        self.assertEqual(avg_blur([1, 2, 3, 4, 5], 2).tolist(), [2, 2.5, 3, 3.5, 4])
