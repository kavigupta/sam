import unittest

import numpy as np
from torch import nn

import torch

from parameterized import parameterized

from modular_splicing.utils.sequence_utils import (
    decode_packed_seq,
    map_to_all_seqs_idx,
    slice_packed_seq,
)


class PackedSeqTest(unittest.TestCase):
    @parameterized.expand([(x,) for x in range(100)])
    def test_pack_inverses(self, seed):
        torch.manual_seed(seed)
        w = np.random.RandomState(seed * 2).choice(12) * 2 + 1
        unpacked = torch.randint(4, size=(10, 51))
        packed = map_to_all_seqs_idx(unpacked, w).numpy()
        for i in range(packed.shape[0]):
            r = decode_packed_seq(packed[i], w)[w // 2 : -w // 2]
            for j in range(r.shape[0]):
                self.assertTrue((r[j] == unpacked[i, j:][:w].numpy()).all())

    @parameterized.expand([(x,) for x in range(100)])
    def test_slice(self, seed):
        rng = np.random.RandomState(seed)
        w = rng.choice(12) * 2 + 1
        packed = rng.choice(4**w)
        unpacked = decode_packed_seq(np.array([packed]), w)

        start, end = sorted(rng.randint(w + 1, size=2))

        unpacked_slice = decode_packed_seq(
            slice_packed_seq(np.array([packed]), start, end), end - start
        )

        self.assertTrue((unpacked[:, start:end] == unpacked_slice).all())
