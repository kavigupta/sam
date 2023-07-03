import itertools
import re

import attr
import h5py
import numpy as np


class MultiH5File:
    """
    Concatenate the files in the list, using higher indices for each
        type as the list progresses.

    E.g., if file one has keys X0, Y0, X1, Y1 and file two has keys X0, Y0, X1, Y1,
        then the resulting file will have keys X0, Y0, X1, Y1, X2, Y2, X3, Y3.

    batch_indices_by_prefix is a dictionary mapping prefixes to the index to use for the
        batch dimension. E.g., if batch_indices_by_prefix is {"X": 0, "Y": 1}, then
        we know that the batch dimension is the first dimension for X and the second
        dimension for Y. (This is common since there is a redundant first dimension
        in spliceai data for some reason). By default, we assume that the batch
        dimension is the first dimension for all keys (i.e., the "X" : 0 above is
        redudant).
    """

    def __init__(
        self, paths, *, equalize_sizes_by_subsampling=False, batch_indices_by_prefix
    ):
        self.files = [h5py.File(p, "r") for p in paths]
        self.key_map = construct_key_map([x.keys() for x in self.files])
        if equalize_sizes_by_subsampling:
            self.index_subset_map = construct_index_subset_map(
                [
                    {
                        k: v.shape[batch_indices_by_prefix.get(compute_prefix(k), 0)]
                        for k, v in f.items()
                    }
                    for f in self.files
                ],
                np.random.choice(2**32),
            )
        else:
            self.index_subset_map = None

        self.batch_indices_by_prefix = batch_indices_by_prefix

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for f in self.files:
            f.close()

    def keys(self):
        return self.key_map.keys()

    def __getitem__(self, key):
        file_index, file_key = self.key_map[key]
        obj = self.files[file_index][file_key]
        if self.index_subset_map is None:
            return obj
        else:
            return IndexedDataWrapper(
                obj,
                self.index_subset_map.get((file_index, file_key), []),
                self.batch_indices_by_prefix.get(compute_prefix(file_key), 0),
            )

    def __contains__(self, key):
        return key in self.key_map

    def __len__(self):
        return len(self.key_map)

    def items(self):
        return ((k, self[k]) for k in self.keys())


@attr.s
class IndexedDataWrapper:
    """
    Allows for indexing into a subset of indices of an object.
    """

    obj = attr.ib()
    index_subset = attr.ib()
    batch_index = attr.ib()

    @property
    def shape(self):
        assert (
            not self.index_subset
            or self.index_subset[-1] < self.obj.shape[self.batch_index]
        )
        return (
            self.obj.shape[: self.batch_index]
            + (len(self.index_subset),)
            + self.obj.shape[self.batch_index + 1 :]
        )

    def __getitem__(self, key):
        obj = self.obj
        obj = obj[:]
        obj = obj.take(self.index_subset, axis=self.batch_index)
        return obj[key]


def prefixes_and_length(keys):
    """
    Given a list of keys of the form LetterNumber, return a tuple of
        (prefixes, length) where prefixes is a tuple of the unique prefixes
        length is the length of the list [0...length-1] that is attached
        to each prefix.

    Produces an error when the keys are not of the form LetterNumber
        or when the length is not the same for each prefix.
    """
    assert len(keys) == len(set(keys)), "keys must be unique"
    prefixes = tuple(sorted({compute_prefix(key) for key in keys}))
    length = len(keys) // len(prefixes)
    expected_keys = {prefix + str(i) for prefix in prefixes for i in range(length)}
    assert expected_keys == set(keys), f"{expected_keys} != {set(keys)}"
    return prefixes, length


KEY_REGEX = "(?P<prefix>[A-Za-z]*)(?P<index>[0-9]*)"


def compute_prefix(key):
    return re.match(KEY_REGEX, key).group("prefix")


def compute_index(key):
    return int(re.match(KEY_REGEX, key).group("index"))


def construct_key_map(keys_each):
    """
    Construct a key map from a list of lists of keys, where each list
        represents the keys into a given file.

    Effectively, concatenates each file's keys by prefix together. E.g.,
        if file one has keys X0, Y0, X1, Y1 and file two has keys X0, Y0, X1, Y1,
        then the resulting file will have keys X0, Y0, X1, Y1, X2, Y2, X3, Y3,
        with X0 and X1 coming from file one and X2 and X3 coming from file two.

    This function would return the dictionary
        {
            "X0": (0, "X0"),
            "Y0": (0, "Y0"),
            "X1": (0, "X1"),
            "Y1": (0, "Y1"),
            "X2": (1, "X0"),
            "Y2": (1, "Y0"),
            "X3": (1, "X1"),
            "Y3": (1, "Y1"),
        }
    """
    prefixes_each, lengths = zip(*[prefixes_and_length(keys) for keys in keys_each])
    assert all(prefixes == prefixes_each[0] for prefixes in prefixes_each), (
        "prefixes must be the same",
        prefixes_each,
    )
    prefixes = prefixes_each[0]
    backmap = {}
    count = 0
    for file_idx, length in enumerate(lengths):
        for prefix in prefixes:
            for i in range(length):
                backmap[prefix + str(count + i)] = (file_idx, prefix + str(i))
        count += length
    return backmap


def construct_index_subset_map(lengths_each, seed):
    """
    Constructs a map from each file index and key to the indices that should be
        used for that key. Randomly sampled such that the number of indices
        is equal across all files.

    lengths_each: a list of dictionaries mapping keys to lengths
    Returns: a dictionary mapping pairs (file index, key) to a list of indices
        within the chunk.
    """
    data_indices_each = [all_data_indices(lengths) for lengths in lengths_each]
    length_to_use = min([len(indices) for indices in data_indices_each])

    rng = np.random.RandomState(seed)
    result = {}
    for i in range(len(data_indices_each)):
        subset = sorted(
            rng.choice(len(data_indices_each[i]), length_to_use, replace=False).tolist()
        )
        subset = [data_indices_each[i][j] for j in subset]
        prefixes, _ = prefixes_and_length(lengths_each[i].keys())
        for chunk_index, group in itertools.groupby(subset, lambda x: x[0]):
            group = list(group)
            for prefix in prefixes:
                result[(i, prefix + str(chunk_index))] = [index for _, index in group]
    return result


def all_data_indices(lengths_by_key):
    """
    Given a dictionary mapping keys to lengths of each data chunk, a list indices for each datapoint.

    Each index is of the form (chunk_key_number, index_within_chunk). The lists should be in
        order, and as such, they can be zippered together.

    E.g., if the dictionary is {"X0": 3, "X1": 5, "Y0": 3, "Y1": 5}, then the returned
        list is [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4)]
    """
    prefixes, num_keys = prefixes_and_length(lengths_by_key.keys())
    result = []
    for i in range(num_keys):
        [length] = {lengths_by_key[prefix + str(i)] for prefix in prefixes}
        for j in range(length):
            result.append((i, j))
    return result
