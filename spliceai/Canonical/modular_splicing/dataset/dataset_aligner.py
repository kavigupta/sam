import h5py
import numpy as np


class DatasetAligner:
    """
    Class for aligning a dataset to a datafile.

    Used to find the correct gene and position within gene for a given
        chunk and within-chunk index in a dataset.

    Effectively, undoes the binning done in `create_dataset.py`
    """

    def __init__(self, dataset_path, datafile_path, sl):
        with h5py.File(dataset_path, "r") as f:
            # dataset_sizes is a list of the size of each chunk in the dataset
            self.dataset_sizes = [f[f"X{i}"].shape[0] for i in range(len(f) // 2)]

        with h5py.File(datafile_path, "r") as f:
            # gets the start and end indices of each gene in the datafile
            start = np.array([int(x.decode("ascii")) for x in f["TX_START"][:]])
            end = np.array([int(x.decode("ascii")) for x in f["TX_END"][:]])
            # computes the size of each gene. This is a ceiling division, as
            # end is inclusive, so the true length is end - start + 1
            self.gene_sizes = (end - start + sl) // sl
            # get the names for each gene
            self.names = [x.decode("ascii") for x in f["NAME"][:]]

    def get_gene_idx(self, i, j):
        return split_bin(self.gene_sizes, to_absolute_index(self.dataset_sizes, i, j))


def bin_starts(bin_sizes):
    """
    Produces an array of the start indices of each bin in a list of bin sizes.

    E.g., bin_starts([3, 2, 1]) = [0, 3, 5, 6]
    """
    return np.array([0, *np.cumsum(bin_sizes)])


def split_bin(bin_sizes, i):
    """
    Get the index of the bin containing the given index, and the index within
        that bin.
    """
    bins = bin_starts(bin_sizes)
    bin = np.where(i >= bins)[0][-1]
    return bin, i - bins[bin]


def to_absolute_index(bin_sizes, i, j):
    """
    Get the absolute index of the given index within a bin.

    E.g., to_absolute_index([3, 2, 1], 1, 1) = 4
    """
    bin_sizes = bin_starts(bin_sizes)
    return bin_sizes[i] + j
