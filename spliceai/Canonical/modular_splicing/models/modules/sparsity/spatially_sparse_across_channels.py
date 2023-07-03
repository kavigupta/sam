import numpy as np
import torch.nn as nn


from .sparsity import SparseLayer
from .spatially_sparse_by_channel import SpatiallySparseByChannel


class SpatiallySparseAcrossChannels(SparseLayer, nn.Module):
    """
    Like SpatiallySparseByChannel but uses the same threshold for all channels.
    """

    def __init__(self, sparsity, num_channels, **kwargs):
        super().__init__()
        del num_channels  # unused
        self.sparse = SpatiallySparseByChannel(sparsity, 1, **kwargs)

    def forward(self, x):
        N, C, L = x.shape
        return self.sparse(x.reshape(N * C, 1, L)).reshape(N, C, L)

    def update_sparsity(self, update_by):
        return self.sparse.update_sparsity(update_by)

    def set_sparsity(self, sparsity):
        self.sparse.set_sparsity(sparsity)

    def get_sparsity(self):
        return self.sparse.get_sparsity()

    def thresholds_numpy(self, num_channels):
        """
        Expand the thresholds to be one per channel.
        """
        [thresh] = self.sparse.thresholds
        thresh = thresh.item()
        threshs = np.array([thresh] * num_channels)
        return threshs

    def motif_index(self, num_channels):
        return
