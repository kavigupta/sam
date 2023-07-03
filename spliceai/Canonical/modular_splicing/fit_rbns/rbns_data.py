import os
from types import SimpleNamespace

import h5py
import numpy as np
import torch
import torch.nn as nn
from modular_splicing.models.motif_models.types import motif_model_types
from modular_splicing.motif_names import get_motif_names

from modular_splicing.utils.construct import construct

import modular_splicing

root_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(modular_splicing.__file__))),
    "data/tiny_binary_rbns_dataset",
)


class RBNSData:
    """
    Class representing RBNS datasets. Does not contain any data in the
    object itself, but rather loads data from disk as needed.

    Parameters
    ----------
    root: the root directory of the dataset
    """

    def __init__(self, root=root_path):
        names = []
        for path in os.listdir(root):
            assert path.startswith("protein_")
            name = path[len("protein_") :]
            name = translate_rbns_data_name(name)
            names.append((name, path))
        names = sorted(names)
        self.names = [n for n, _ in names]
        self.paths = {n: os.path.join(root, p, "rbns_train.h5") for n, p in names}

    def path(self, name, is_test):
        """
        Get the path to the data for a given protein.

        If `is_test` is True, the test set is returned. Otherwise, the
        training set is returned.
        """
        res = self.paths[name]
        if is_test:
            res = res.replace("train", "test")
        return res

    def length(self, name, is_test=False):
        """
        Get the number of examples in the dataset.
        """
        with h5py.File(self.path(name, is_test), "r") as f:
            return f["X"].shape[0]

    def data(self, name, idxs=slice(None, None), is_test=False):
        """
        Return the data for a given protein.

        Parameters
        ----------
        name: the name of the protein
        idxs: the indices to return. Defaults to all indices.
        is_test: whether to return the test set. Defaults to False.
        """
        with h5py.File(self.path(name, is_test), "r") as f:
            X, Y, L = [f[c][idxs] for c in "XYL"]
        L = np.minimum(L, 40)
        X = X[:, : L.max()]
        mask = np.arange(X.shape[1])[None] < L[:, None]
        return SimpleNamespace(x=X, y=Y, l=L, mask=mask)

    def batched_data(self, name, batch_size, *, seed, is_test=False):
        """
        Return an iterator over batches of data for a given protein.

        Parameters
        ----------
        name: the name of the protein
        batch_size: the batch size
        seed: the random seed to use in shuffling the data
        is_test: whether to return the test set. Defaults to False.
        """
        overall = self.data(name, is_test=is_test)
        idxs = np.arange(self.length(name, is_test=is_test))
        np.random.RandomState(seed).shuffle(idxs)
        for idx in range((idxs.shape[0] + batch_size - 1) // batch_size):
            start = idx * batch_size
            these = idxs[start : start + batch_size]
            these.sort()
            yield SimpleNamespace(**{k: v[these] for k, v in vars(overall).items()})


def translate_rbns_data_name(x):
    """
    Some of the names in the folder are represented by indices, into the list
    of RBNS motif names. These are offset by 2, because the original list of
    names had 3P and 5P at the beginning.

    The rest are represented by a string name.
    """
    rbns_motif_names = ["3P", "5P"] + get_motif_names("rbns")
    try:
        return rbns_motif_names[int(x)]
    except ValueError:
        return x


class MotifModelBasedClassifer(nn.Module):
    """
    Use a given underlying motif model as a classifier
        of RBNS data.

    The prediction produced is the maximum of the motif scores
    across the sequence, treated as atanh logits,
    clamped to the range [0.00001, 1].
    """

    def __init__(self, motif_model_spec):
        super().__init__()
        self.motif_model = construct(
            motif_model_types(),
            motif_model_spec,
            input_size=4,
            channels=20,
            num_motifs=2,
        )

    def forward(self, x):
        motifs = self.motif_model(x)
        if isinstance(motifs, dict):
            motifs = motifs["motifs"]
        motifs = motifs[:, :, 0]
        raw_yp, _ = motifs.max(-1)
        yp = torch.clamp(raw_yp.tanh(), 1e-5, 1)
        return yp

    def loss(self, x, y):
        x = torch.tensor(x.astype(np.float32)).cuda()
        y = torch.tensor(y.astype(np.float32)).cuda()
        yp = self(x)
        return yp, nn.BCELoss()(yp, y)
