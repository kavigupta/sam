import unittest

import h5py
import numpy as np
import tqdm.auto as tqdm

from modular_splicing.dataset.dataset_aligner import (
    DatasetAligner,
    bin_starts,
    split_bin,
    to_absolute_index,
)
from modular_splicing.utils.sequence_utils import draw_bases


class DatasetAlignerTest(unittest.TestCase):
    def setUp(self):
        self.aligner = DatasetAligner(
            dataset_path="dataset_train_all.h5",
            datafile_path="datafile_train_all.h5",
            sl=5000,
        )
        self.cl_max = 10_000
        self.sl = 5000

    def check_index(self, dset, dfil, i, j):
        gene_idx, within_gene = self.aligner.get_gene_idx(i, j)

        # clip off the CL
        actual_seq_contents = dfil["SEQ"][gene_idx][
            self.cl_max // 2 : -self.cl_max // 2
        ]
        actual_seq_contents = actual_seq_contents.upper()

        if dfil["STRAND"][gene_idx] == b"-":
            actual_seq_contents = actual_seq_contents[::-1]
            actual_seq_contents = actual_seq_contents.decode("utf-8")
            actual_seq_contents = actual_seq_contents.translate(
                str.maketrans("ATCG", "TAGC")
            )
            actual_seq_contents = actual_seq_contents.encode("utf-8")

        # get the targeted segment as suggested by the datafile and aligner's indices
        fil_seq = actual_seq_contents[
            within_gene * self.sl : (within_gene + 1) * self.sl
        ]

        # get the targeted segment as suggested by the dataset
        set_seq = dset[f"X{i}"][j][self.cl_max // 2 : -self.cl_max // 2]
        set_seq = "".join(draw_bases(set_seq)).encode("utf-8")
        # pad out fil seq to match set seq
        fil_seq = fil_seq + b"N" * (len(set_seq) - len(fil_seq))
        self.assertEqual(fil_seq, set_seq)

    def test_aligner(self):
        rng = np.random.RandomState(0)
        for _ in tqdm.trange(1000):
            with h5py.File("dataset_train_all.h5", "r") as dset:
                with h5py.File("datafile_train_all.h5", "r") as dfil:
                    i = rng.choice(len(dset) // 2)
                    j = rng.choice(len(dset[f"X{i}"]))
                    self.check_index(dset, dfil, i, j)

    def test_bin_starts(self):
        self.assertEqual(bin_starts([1, 2, 3]).tolist(), [0, 1, 3, 6])

    def test_split_bin(self):
        self.assertEqual(split_bin([1, 2, 3], 0), (0, 0))
        self.assertEqual(split_bin([1, 2, 3], 1), (1, 0))
        self.assertEqual(split_bin([1, 2, 3], 2), (1, 1))
        self.assertEqual(split_bin([1, 2, 3], 3), (2, 0))
        self.assertEqual(split_bin([1, 2, 3], 4), (2, 1))
        self.assertEqual(split_bin([1, 2, 3], 5), (2, 2))

    def test_to_absolute_index(self):
        self.assertEqual(to_absolute_index([1, 2, 3], 0, 0), 0)
        self.assertEqual(to_absolute_index([1, 2, 3], 1, 0), 1)
        self.assertEqual(to_absolute_index([1, 2, 3], 1, 1), 2)
        self.assertEqual(to_absolute_index([1, 2, 3], 2, 0), 3)
        self.assertEqual(to_absolute_index([1, 2, 3], 2, 1), 4)
        self.assertEqual(to_absolute_index([1, 2, 3], 2, 2), 5)
