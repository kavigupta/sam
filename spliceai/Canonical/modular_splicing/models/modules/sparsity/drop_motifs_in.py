import torch.nn as nn

from modular_splicing.models.modules.sparsity.sparsity import SparseLayer


class DropMotifsIn(SparseLayer, nn.Module):
    """
    Drop the given motifs after running the given sparse layer.
    """

    def __init__(self, sparse_layer, dropped_motifs):
        super().__init__()
        self.sparse_layer = sparse_layer
        self.dropped_motifs = dropped_motifs

    def forward(self, x):
        x = self.sparse_layer(x)
        x[:, self.dropped_motifs, :] = 0
        return x

    def update_sparsity(self, update_by):
        return self.sparse_layer.update_sparsity(update_by)

    def set_sparsity(self, sparsity):
        self.sparse_layer.set_sparsity(sparsity)

    def get_sparsity(self):
        return self.sparse_layer.get_sparsity()

    def thresholds_numpy(self, num_channels):
        mask = [x not in self.dropped_motifs for x in range(num_channels)]
        return self.sparse_layer.thresholds_numpy(num_channels)[mask]

    def motif_index(self, num_motifs):
        dropped = set(self.dropped_motifs)
        mapping = {}
        for i in range(num_motifs):
            if i in dropped:
                continue
            mapping[i] = len(mapping)
        return mapping
