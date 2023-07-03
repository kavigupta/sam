import os
import tempfile
import unittest

import h5py
import numpy as np
from modular_splicing.dataset.h5_dataset import H5Dataset

from modular_splicing.utils.multi_h5_file import (
    MultiH5File,
    all_data_indices,
    construct_index_subset_map,
    construct_key_map,
    prefixes_and_length,
)


class TestPrefixesAndLength(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(
            (("X", "Y"), 4),
            prefixes_and_length(["X0", "Y0", "X1", "Y1", "X2", "Y2", "X3", "Y3"]),
        )

    def test_with_one_prefix_only(self):
        self.assertEqual(
            (("X",), 4),
            prefixes_and_length(["X0", "X1", "X2", "X3"]),
        )

    def test_with_multi_char_prefixes(self):
        self.assertEqual(
            (("X", "YY"), 2),
            prefixes_and_length(["X0", "YY0", "X1", "YY1"]),
        )

    def test_error_skipped_index(self):
        self.assertRaises(
            AssertionError,
            prefixes_and_length,
            ["X0", "Y0", "X1", "Y1", "X3", "Y3"],
        )

    def test_error_missing_index(self):
        self.assertRaises(
            AssertionError,
            prefixes_and_length,
            ["X0", "Y0", "X1", "Y1", "X2"],
        )


class TestConstructKeyMap(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(
            {
                "X0": (0, "X0"),
                "Y0": (0, "Y0"),
                "X1": (0, "X1"),
                "Y1": (0, "Y1"),
                "X2": (1, "X0"),
                "Y2": (1, "Y0"),
                "X3": (1, "X1"),
                "Y3": (1, "Y1"),
            },
            construct_key_map([["X0", "Y0", "X1", "Y1"], ["X0", "Y0", "X1", "Y1"]]),
        )

    def test_three(self):
        self.assertEqual(
            {
                "X0": (0, "X0"),
                "Y0": (0, "Y0"),
                "X1": (0, "X1"),
                "Y1": (0, "Y1"),
                "X2": (1, "X0"),
                "Y2": (1, "Y0"),
                "X3": (1, "X1"),
                "Y3": (1, "Y1"),
                "X4": (2, "X0"),
                "Y4": (2, "Y0"),
            },
            construct_key_map(
                [["X0", "Y0", "X1", "Y1"], ["X0", "Y0", "X1", "Y1"], ["X0", "Y0"]]
            ),
        )

    def test_different_lengths_each(self):
        actual = construct_key_map(
            [
                ["X0", "Y0"],
                ["X0", "Y0", "X1", "Y1", "X2", "Y2"],
                ["X0", "Y0", "X1", "Y1"],
            ]
        )
        expected = {
            "X0": (0, "X0"),
            "Y0": (0, "Y0"),
            "X1": (1, "X0"),
            "Y1": (1, "Y0"),
            "X2": (1, "X1"),
            "Y2": (1, "Y1"),
            "X3": (1, "X2"),
            "Y3": (1, "Y2"),
            "X4": (2, "X0"),
            "Y4": (2, "Y0"),
            "X5": (2, "X1"),
            "Y5": (2, "Y1"),
        }
        self.assertEqual(expected, actual)


class TestAllDataIndices(unittest.TestCase):
    def basic_test(self):
        self.assertEqual(
            all_data_indices({"X0": 3, "X1": 5, "Y0": 3, "Y1": 5}),
            [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4)],
        )


class TestConstructIndexSubsetMap(unittest.TestCase):
    def one_chunk_test(self):
        counts = np.zeros(5)
        for seed in range(1000):
            files = [{"X0": 2, "Y0": 2}, {"X0": 5, "Y0": 5}]
            result = construct_index_subset_map(files, seed=seed)
            self.assertEqual(
                result.keys(), {(0, "X0"), (0, "Y0"), (1, "X0"), (1, "Y0")}
            )
            self.assertEqual(result[(0, "X0")], [0, 1])
            self.assertEqual(result[(0, "Y0")], [0, 1])
            self.assertEqual(result[(1, "X0")], result[(1, "Y0")])
            self.assertEqual(result[(1, "X0")], sorted(set(result[(1, "X0")])))
            self.assertEqual(len(result[(1, "X0")]), 2)
            self.assertTrue(set(result[(1, "X0")]).issubset({0, 1, 2, 3, 4}))
            counts[result[(1, "X0")]] += 1

        counts = counts / counts.mean()
        # each should be about 1/5th
        for i in range(5):
            self.assertGreater(counts[i], 0.95)
            self.assertLess(counts[i], 1.05)

    def multi_chunk_test(self):
        counts = np.zeros(10)
        for seed in range(10_000):
            files = [
                {"X0": 2, "Y0": 2, "X1": 4, "Y1": 4},
                {"X0": 5, "Y0": 5, "X1": 5, "Y1": 5},
            ]
            result = construct_index_subset_map(files, seed=seed)
            self.assertEqual(
                result.keys(),
                {
                    (0, "X0"),
                    (0, "Y0"),
                    (1, "X0"),
                    (1, "Y0"),
                    (0, "X1"),
                    (0, "Y1"),
                    (1, "X1"),
                    (1, "Y1"),
                },
            )
            self.assertEqual(result[(0, "X0")], [0, 1])
            self.assertEqual(result[(0, "Y0")], [0, 1])
            self.assertEqual(result[(0, "X1")], [0, 1, 2, 3])
            self.assertEqual(result[(0, "Y1")], [0, 1, 2, 3])
            self.assertEqual(result[(1, "X0")], result[(1, "Y0")])
            self.assertEqual(result[(1, "X0")], sorted(set(result[(1, "X0")])))
            self.assertTrue(set(result[(1, "X0")]).issubset({0, 1, 2, 3, 4}))
            self.assertEqual(result[(1, "X1")], result[(1, "Y1")])
            self.assertEqual(result[(1, "X1")], sorted(set(result[(1, "X1")])))
            self.assertTrue(set(result[(1, "X1")]).issubset({0, 1, 2, 3, 4}))
            self.assertEqual(len(result[(1, "X0")]) + len(result[(1, "X1")]), 6)
            counts[result[(1, "X0")]] += 1
            counts[[x + 5 for x in result[(1, "X1")]]] += 1

        counts = counts / counts.mean()
        # each should be about 1/10th
        for i in range(10):
            self.assertGreater(counts[i], 0.95)
            self.assertLess(counts[i], 1.05)


SL = 50
CL = 10
C = 4


class MultiH5FileTest(unittest.TestCase):
    def create_h5_file(self, f, *length_each, start_idx):
        idxs_each = []
        for i, length in enumerate(length_each):
            idxs = np.arange(start_idx, start_idx + length)
            f[f"X{i}"] = np.zeros((length, SL + CL, C)) + idxs[:, None, None]
            f[f"Y{i}"] = -(np.zeros((1, length, SL, C)) + idxs[:, None, None])
            start_idx += length
            idxs_each.append(idxs)
        return start_idx, np.concatenate(idxs_each).tolist()

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        start_idx = 0
        with h5py.File(os.path.join(self.directory, "A.h5"), "w") as f:
            start_idx, self.a_ids = self.create_h5_file(f, 2, 3, start_idx=start_idx)

        with h5py.File(os.path.join(self.directory, "B.h5"), "w") as f:
            start_idx, self.b_ids = self.create_h5_file(f, 2, 1, start_idx=start_idx)

    def contents(self, f, key):
        res = f[key]
        assert not isinstance(res, np.ndarray)
        assert res.shape == res[:].shape
        res = res[:]
        if key.startswith("Y"):
            assert res.shape[0] == 1
            res = res[0]
        assert isinstance(res, np.ndarray)
        assert res.shape[1:] == (CL * key.startswith("X") + SL, C)
        assert (res[:, 0, 0][:, None, None] == res).all()
        return res[:, 0, 0].tolist()

    def test_concatenation(self):
        with MultiH5File(
            [
                os.path.join(self.directory, "A.h5"),
                os.path.join(self.directory, "B.h5"),
            ],
            batch_indices_by_prefix={"Y": 1},
        ) as f:
            self.assertEqual(len(f), 8)
            self.assertEqual(self.contents(f, "X0"), self.a_ids[:2])
            self.assertEqual(self.contents(f, "Y0"), [-x for x in self.a_ids[:2]])
            self.assertEqual(self.contents(f, "X1"), self.a_ids[2:5])
            self.assertEqual(self.contents(f, "Y1"), [-x for x in self.a_ids[2:5]])
            self.assertEqual(self.contents(f, "X2"), self.b_ids[:2])
            self.assertEqual(self.contents(f, "Y2"), [-x for x in self.b_ids[:2]])
            self.assertEqual(self.contents(f, "X3"), self.b_ids[2:3])
            self.assertEqual(self.contents(f, "Y3"), [-x for x in self.b_ids[2:3]])

    def test_equalization(self):
        np.random.seed(0)
        counts = np.zeros(5)
        for _ in range(1000):
            with MultiH5File(
                [
                    os.path.join(self.directory, "A.h5"),
                    os.path.join(self.directory, "B.h5"),
                ],
                equalize_sizes_by_subsampling=True,
                batch_indices_by_prefix={"Y": 1},
            ) as f:
                self.assertEqual(len(f), 8)
                x0_vals = self.contents(f, "X0")
                self.assertEqual(x0_vals, [-x for x in self.contents(f, "Y0")])
                x1_vals = self.contents(f, "X1")
                self.assertEqual(x1_vals, [-x for x in self.contents(f, "Y1")])

                self.assertEqual(self.contents(f, "X2"), self.b_ids[:2])
                self.assertEqual(self.contents(f, "Y2"), [-x for x in self.b_ids[:2]])
                self.assertEqual(self.contents(f, "X3"), self.b_ids[2:3])
                self.assertEqual(self.contents(f, "Y3"), [-x for x in self.b_ids[2:3]])

                first_vals = [int(x) for x in x0_vals + x1_vals]

                self.assertEqual(len(first_vals), len(self.b_ids))
                self.assertEqual(sorted(set(first_vals)), first_vals)

                counts[first_vals] += 1

        counts = counts / counts.mean()
        # each should be about 1/5th
        for i in range(5):
            self.assertGreater(counts[i], 0.9)
            self.assertLess(counts[i], 1.1)

    def dataset(self, **kwargs):
        return H5Dataset(
            datapoint_extractor_spec=dict(
                type="BasicDatapointExtractor", run_argmax=False
            ),
            post_processor_spec=dict(type="IdentityPostProcessor"),
            path=[
                os.path.join(self.directory, "A.h5"),
                os.path.join(self.directory, "B.h5"),
            ],
            cl=CL,
            cl_max=CL,
            sl=SL,
            iterator_spec=dict(
                type="FastIter", shuffler_spec=dict(type="DoNotShuffle")
            ),
            **kwargs,
        )

    def test_end_to_end_without_equalization(self):
        data = self.dataset()
        self.assertEqual(len(data), len(self.a_ids) + len(self.b_ids))
        xs = []
        for it in data:
            x, y = it["inputs"]["x"], it["outputs"]["y"]
            self.assertEqual(x.shape, (SL + CL, C))
            self.assertEqual(y.shape, (SL, C))
            self.assertTrue((x == x[0, 0]).all())
            self.assertTrue((y == -x[0, 0]).all())
            xs.append(int(x[0, 0]))
        self.assertEqual(xs, list(range(8)))

    def test_end_to_end_with_equalization(self):
        counts = np.zeros(5)
        np.random.seed(0)
        for _ in range(1000):
            data = self.dataset(equalize_sizes_by_subsampling=True)
            self.assertEqual(len(data), 2 * len(self.b_ids))
            xs = []
            for it in data:
                x, y = it["inputs"]["x"], it["outputs"]["y"]
                self.assertEqual(x.shape, (SL + CL, C))
                self.assertEqual(y.shape, (SL, C))
                self.assertTrue((x == x[0, 0]).all())
                self.assertTrue((y == -x[0, 0]).all())
                xs.append(int(x[0, 0]))
            self.assertEqual(xs, sorted(set(xs)))
            xs = set(xs)
            self.assertTrue(set(self.b_ids).issubset(xs))
            xs = xs - set(self.b_ids)
            self.assertTrue(xs.issubset(set(self.a_ids)))
            counts[list(xs)] += 1
        counts = counts / counts.mean()
        # each should be about 1/5th
        for i in range(5):
            self.assertGreater(counts[i], 0.9)
            self.assertLess(counts[i], 1.1)
