import torch.nn as nn

from modular_splicing.models.modules.sparsity.sparsity import SparseLayer


class NoSparsity(SparseLayer, nn.Module):
    """
    Acts like a sparse layer but does not actually enforce sparsity in any way.

    Pretends to have sparsity to keep code that interacts with it happy.
    """

    def __init__(
        self,
        sparsity,
        num_channels,
        momentum=0.1,
    ):
        super().__init__()
        self.sparsity = sparsity

    def update_sparsity(self, update_by):
        print(f"Originally dropping {self.sparsity}")
        self.sparsity = 1 - (1 - self.sparsity) * update_by
        print(f"Now dropping {self.sparsity}")

    def set_sparsity(self, sparsity):
        self.sparsity = sparsity

    def get_sparsity(self):
        return self.sparsity

    def forward(self, x):
        return x

    def thresholds_numpy(self, num_channels):
        raise RuntimeError("Not reasonable to get thresholds for NoSparsity")

    def motif_index(self, num_channels):
        return
