import attr
from cached_property import cached_property
from permacache import stable_hash

import numpy as np


def run_length_encoding(inarray):
    """
    run length encoding. Partial credit to R rle function.
    Multi datatype arrays catered for including non Numpy
    returns: tuple (runlengths, startpositions, values)

    From: https://stackoverflow.com/a/32681075
    """
    ia = np.asarray(inarray)  # force numpy
    n = len(ia)
    if n == 0:
        return (None, None, None)
    else:
        y = ia[1:] != ia[:-1]  # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)  # must include last element posi
        z = np.diff(np.append(-1, i))  # run lengths
        p = np.cumsum(np.append(0, z))[:-1]  # positions
        return (z, p, ia[i])


def add_cl_in(ys, cl, pad_value=0):
    """
    Add in a context length to the given sequence, tiling the
    given sequence to fill the context length.

    See test for examples.

    Parameters:
    ys: (N, L)
        the sequences
    cl: int
        the context length
    pad_value: int
        the value to pad with

    Returns: (N, L + cl)
        the sequences with the context length added in
    """
    assert cl % 2 == 0
    radius = cl // 2
    assert radius <= ys.shape[1]
    trim = ys.shape[1] - radius

    padding = np.zeros_like(ys[:1])
    padding[:] = pad_value

    left = np.concatenate([padding, ys[:-1]], axis=0)
    right = np.concatenate([ys[1:], padding], axis=0)

    ys = np.concatenate([left, ys, right], axis=1)

    return ys[:, trim : ys.shape[1] - trim]


@attr.s
class Sparse:
    """
    Represents a tensor x with the following fields
        - shape: x.shape
        - where: np.where(x)
        - values: x[where]
        - dtype: x.dtype

    This is useful for storing sparse tensors in a compact way. It is not meant to be used for
    computations.
    """

    shape = attr.ib()
    where = attr.ib()
    values = attr.ib()
    dtype = attr.ib()

    @classmethod
    def of(cls, tensor):
        """
        Computes the sparse representation of a tensor.
        """
        where = np.where(tensor)
        return cls(tensor.shape, where, tensor[where], tensor.dtype)

    @property
    def original(self):
        """
        Computes the original tensor from the sparse representation.
        """
        out = np.zeros(self.shape, self.dtype)
        out[self.where] = self.values
        return out

    @property
    def size(self):
        """
        Equivalent of self.original.size
        """
        result = 1
        for d in self.shape:
            result *= d
        return result

    @cached_property
    def stable_hash(self):
        """
        Computes a stable hash of the sparse representation.
        """
        return stable_hash(self, fast_bytes=True)
