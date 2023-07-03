import torch.nn as nn


class NoPresparseNormalizer(nn.Module):
    """
    Do not normalize the input to the sparse layer.
    """

    def __init__(self, num_motifs, affine):
        super().__init__()
        self.num_motifs = num_motifs

    def forward(self, x):
        return x

    @property
    def num_features(self):
        return self.num_motifs


class BasicPresparseNormalizer(nn.Module):
    """
    Do a batch norm on the input to the sparse layer.

    If `affine` is True, then the batch norm will learn a scale and bias
    parameter. Otherwise, it will just normalize the input.

    In practice, this means that different channels
        can learn different sparsities.
    """

    def __init__(self, num_motifs, affine):
        super().__init__()
        self.norm = nn.BatchNorm1d(num_motifs, affine=affine)

    def forward(self, x):
        return self.norm(x)

    @property
    def num_features(self):
        return self.norm.num_features
