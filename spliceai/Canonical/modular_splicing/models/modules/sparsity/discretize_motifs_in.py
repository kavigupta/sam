import torch
import torch.nn as nn

from .sparsity import SparseLayer


class DiscretizeMotifsIn(SparseLayer, nn.Module):
    """
    Binarize the motifs after running the given sparse layer.
    """

    def __init__(self, sparse_layer):
        super().__init__()
        self.sparse_layer = sparse_layer

    def forward(self, x):
        with torch.no_grad():
            x = self.sparse_layer(x)
        x = (x != 0).float()
        return x

    def update_sparsity(self, update_by):
        return self.sparse_layer.update_sparsity(update_by)

    def set_sparsity(self, sparsity):
        self.sparse_layer.set_sparsity(sparsity)

    def get_sparsity(self):
        return self.sparse_layer.get_sparsity()

    def thresholds_numpy(self, num_channels):
        return self.sparse_layer.thresholds_numpy(num_channels)

    def motif_index(self, num_motifs):
        return self.sparse_layer.motif_index(num_motifs)
