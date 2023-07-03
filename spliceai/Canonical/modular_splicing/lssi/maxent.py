import attr

import more_itertools
from permacache import permacache, stable_hash

import tqdm.auto as tqdm
import numpy as np

from .utils import second_half_of_spliceai_cached


@attr.s
class FullTable:
    values = attr.ib()
    indices = attr.ib()

    @indices.validator
    def check(self, _attribute, _value):
        assert len(self.values.shape) == 1
        assert 4 ** len(self.indices) == self.values.size, str(
            (4 ** len(self.indices), self.values.size)
        )

    @classmethod
    def from_file(cls, path, indices):
        with open(path) as f:
            values = np.log([float(x) for x in f])
        return cls(values, indices)

    @classmethod
    def from_npz(cls, path_prefix, indices, denom_adj=0):
        numer = np.load(path_prefix + ".npy")
        denom = np.load(path_prefix + "_0.npy")
        assert numer.shape == denom.shape
        values = np.log(numer) - np.log(denom + denom_adj / denom.size)
        return cls(values, indices)

    def invert(self):
        return FullTable(-self.values, self.indices)

    @property
    def place_vals(self):
        return 4 ** np.arange(len(self.indices), dtype=np.long)[::-1]

    def scan(self, x, cl=5000):
        assert cl * 2 > len(self.indices)
        x = x.argmax(-1)
        assert self.indices == sorted(self.indices)
        batch_idxs, locations, loc_idxs = np.meshgrid(
            np.arange(x.shape[0], dtype=np.long),
            np.arange(x.shape[1] - max(self.indices) - 1, dtype=np.long),
            self.indices,
            indexing="ij",
        )
        idxs = x[batch_idxs, loc_idxs + locations]
        predictions = self.values[idxs @ self.place_vals]
        return predictions[:, cl : x.shape[1] - cl]


def splice_predictions(table, x, **kwargs):
    return np.sum([t.scan(x, **kwargs) for t in table], axis=0)


@permacache(
    "modular_splicing/lssi/maxent/run_maxent_on_data_2",
    key_function=dict(m=stable_hash),
)
def run_maxent_on_data(m):
    xs, _ = second_half_of_spliceai_cached()
    return np.concatenate(
        [
            splice_predictions(m, np.array(x), cl=50 // 2)
            for x in tqdm.tqdm(list(more_itertools.chunked(xs, 1000)))
        ]
    )
