"""
Various techniques for shuffling items.
"""

import numpy as np


class UnseededShuffler:
    def shuffle(self, xs):
        np.random.shuffle(xs)


class SeededShuffler:
    def __init__(self, seed):
        assert seed is not None
        self.rng = np.random.RandomState(seed)

    def shuffle(self, items):
        self.rng.shuffle(items)


class DoNotShuffle:
    def shuffle(self, items):
        pass


def shuffler_types():
    return dict(
        UnseededShuffler=UnseededShuffler,
        SeededShuffler=SeededShuffler,
        DoNotShuffle=DoNotShuffle,
    )
