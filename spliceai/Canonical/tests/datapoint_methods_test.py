import unittest

import numpy as np

from modular_splicing.dataset.h5_dataset import clip_datapoint, modify_sl


class ClipDatapointTest(unittest.TestCase):
    def test_clip_datapoint_basic(self):
        arr = np.random.RandomState(0).randint(0, 4, size=(1000, 13, 20))
        self.assertTrue((clip_datapoint(arr, CL=10, CL_max=20) == arr[5:-5]).all())
        self.assertTrue((clip_datapoint(arr, CL=20, CL_max=20) == arr).all())
        self.assertTrue((clip_datapoint(arr, CL=0, CL_max=20) == arr[10:-10]).all())


class ModifySLTest(unittest.TestCase):
    def test_modify_sl_input(self):
        arr = np.arange(12)
        self.assertEqual(
            [x.tolist() for x in modify_sl(arr, SL=5, CL=2, is_output=False)],
            [[0, 1, 2, 3, 4, 5, 6], [5, 6, 7, 8, 9, 10, 11]],
        )
        self.assertEqual(
            [x.tolist() for x in modify_sl(arr, SL=2, CL=2, is_output=False)],
            [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7], [6, 7, 8, 9], [8, 9, 10, 11]],
        )
        self.assertEqual(
            [x.tolist() for x in modify_sl(arr, SL=2, CL=4, is_output=False)],
            [
                [0, 1, 2, 3, 4, 5],
                [2, 3, 4, 5, 6, 7],
                [4, 5, 6, 7, 8, 9],
                [6, 7, 8, 9, 10, 11],
            ],
        )
