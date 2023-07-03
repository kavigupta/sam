import numpy as np
import torch
import torch.nn as nn

from .sparsity import SparseLayer, sparse_layer_types

from modular_splicing.utils.construct import construct


class ParallelSpatiallySparse(SparseLayer, nn.Module):
    """
    Several different sparse layers, applied to different slices of the input.

    Parameters
    ----------
    sparse_specs : list of specifications for sparse layers
    num_channels_each : list of ints
        Number of channels to apply each sparse layer to.
    num_channels : int
        Total number of channels in the input. Must be the sum of num_channels_each.
    update_indices : list of ints
        Indices of sparse layers to update when set/update_sparsity is called
    get_index : int
        Index of sparse layer to get sparsity from when get_sparsity is called
    **kwargs : extra arguments to pass to sparse layers at construction time
    """

    def __init__(
        self,
        sparse_specs,
        num_channels_each,
        num_channels,
        update_indices,
        get_index,
        **kwargs,
    ):
        super().__init__()
        self.num_channels_each = num_channels_each
        self.slices = []
        start = 0
        for size in num_channels_each:
            self.slices.append(slice(start, start + size))
            start += size
        assert start == num_channels
        self.sub_sparse_layers = nn.ModuleList(
            [
                construct(sparse_layer_types(), spec, num_channels=nm, **kwargs)
                for spec, nm in zip(sparse_specs, num_channels_each)
            ]
        )
        self.update_indices = update_indices
        self.get_index = get_index

    def update_sparsity(self, update_by):
        for i in self.update_indices:
            self.sub_sparse_layers[i].update_sparsity(update_by)

    def set_sparsity(self, sparsity):
        for i in self.update_indices:
            self.sub_sparse_layers[i].set_sparsity(sparsity)

    def get_sparsity(self):
        return self.sub_sparse_layers[self.get_index].get_sparsity()

    def forward(self, x):
        """
        Run each sub-sparse layer on its corresponding slice of the input.
        """
        sparsified = [
            m(x[:, xidxs, :]) for m, xidxs in zip(self.sub_sparse_layers, self.slices)
        ]
        return torch.cat(sparsified, dim=1)

    def thresholds_numpy(self, num_channels):
        """
        Concatenates the thresholds from each sub-sparse layer.
        """
        return np.concatenate(
            [
                layer.thresholds_numpy(nce)
                for nce, layer in zip(self.num_channels_each, self.sub_sparse_layers)
            ]
        )

    def motif_index(self, num_channels):
        """
        Only works for the case where each sub-sparse layer doesn't drop any motifs.
        """
        if all(
            layer.motif_index(nce) is None
            for nce, layer in zip(self.num_channels_each, self.sub_sparse_layers)
        ):
            return None
        raise RuntimeError("Not implemented")
