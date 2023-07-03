import torch
import torch.nn as nn

from .sparsity import SparseLayer
from .spatially_sparse_across_channels import SpatiallySparseAcrossChannels


class SpatiallySparseAcrossChannelsDropMotifs(SparseLayer, nn.Module):
    """
    Like `SpatiallySparseAcrossChannels`, but drops motifs every time
        the density is updated across a boundary of `sparse_drop_motif_frequency`.

    Arguments (that aren't passed to `SpatiallySparseAcrossChannels`):
        sparse_drop_motif_frequency: the frequency at which motifs are dropped.
            For example, if `sparse_drop_motif_frequency` is 0.5, then motifs
            are dropped every time the density is updated to be less than
            0.5**K * `density`.
        zero_count_momentum: the momentum with which to update the zero counts
    """

    def __init__(
        self,
        sparsity,
        *,
        num_channels,
        sparse_drop_motif_frequency,
        zero_count_momentum=0.1,
        **kwargs,
    ):
        super().__init__()
        self.sparse = SpatiallySparseAcrossChannels(sparsity, 1, **kwargs)
        self.initial_sparsity = sparsity
        self.sparse_decrease_frequency = sparse_drop_motif_frequency
        self.dropped = []
        self.zero_counts = torch.nn.parameter.Parameter(
            torch.zeros(num_channels), requires_grad=False
        )
        self.zero_count_momentum = zero_count_momentum

    def forward(self, x):
        x[:, self.dropped, :] = 0
        x = self.sparse(x)
        # update the zero counts
        if self.training:
            zero_count_estimate = (x == 0).float().mean(0).mean(1)
            self.zero_counts.data = (
                self.zero_counts.data * (1 - self.zero_count_momentum)
                + self.zero_count_momentum * zero_count_estimate
            )
        return x

    def enough_dropped(self):
        """
        Returns True if the density is greater than the next boundary.
        """
        init_nonzero = 1 - self.initial_sparsity
        next_nonzero_bar = init_nonzero * self.sparse_decrease_frequency ** (
            len(self.dropped) + 1
        )
        return (1 - self.get_sparsity()) > next_nonzero_bar

    def update_sparsity(self, update_by):
        self.sparse.update_sparsity(update_by)
        self._on_sparsity_update()

    def set_sparsity(self, sparsity):
        self.sparse.set_sparsity(sparsity)
        self._on_sparsity_update()

    def _on_sparsity_update(self):
        _, order = self.zero_counts.sort()
        for motif in reversed(order):
            if self.enough_dropped():
                break
            motif = motif.item()
            if motif not in self.dropped:
                print("dropping motif", motif)
                self.dropped.append(motif)

    def get_sparsity(self):
        return self.sparse.get_sparsity()

    def motif_index(self, num_motifs):
        dropped = set(self.dropped)
        mapping = {}
        for i in range(self.zero_counts.shape[0]):
            if i in dropped:
                continue
            mapping[i] = len(mapping)
        return mapping

    def thresholds_numpy(self, num_channels):
        return self.thresholds_numpy(num_channels - len(self.dropped))
