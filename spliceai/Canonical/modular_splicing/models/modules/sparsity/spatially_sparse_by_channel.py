import torch
import torch.nn as nn

from modular_splicing.utils.construct import construct
from modular_splicing.models.modules.activations import activation_types

from .sparsity import SparseLayer


class SpatiallySparseByChannel(SparseLayer, nn.Module):
    """
    Spatially sparse by channel.

    This is a sparse layer that enforces the same sparisty to be the
        same in each channel.

    Thresholds are updated using an exponential moving average with momentum `self.momentum`.
    They start off at 0, and are updated to the `sparsity`th quantile the layer.

    Parameters
    ----------
    sparsity : float
        Initial fraction of outputs to drop.
    num_channels : int
        Number of channels in the layer. Corresponds to the number of motifs.
    momentum : float
        Momentum for updating the thresholds.
    """

    def __init__(
        self,
        sparsity,
        num_channels,
        momentum=0.1,
        relu_spec=dict(type="ReLU"),
    ):
        super().__init__()
        self.sparsity = sparsity
        self.thresholds = torch.nn.parameter.Parameter(
            torch.zeros(num_channels), requires_grad=False
        )
        self.momentum = momentum
        self.relu_module = construct(activation_types(), relu_spec)

    def update_sparsity(self, update_by):
        print(f"Originally dropping {self.sparsity}")
        self.sparsity = 1 - (1 - self.sparsity) * update_by
        print(f"Now dropping {self.sparsity}")

    def set_sparsity(self, sparsity):
        self.sparsity = sparsity

    def get_sparsity(self):
        return self.sparsity

    def forward(self, x):
        # necessary to ensure we don't accidentally run an obsolete model in the wrong mode
        assert getattr(self, "discontinuity_mode", "subtraction") == "subtraction"
        assert getattr(self, "by_magnitude", False) is False
        N, C, L = x.shape

        if self.training:
            to_drop = max(1, int(N * L * self.sparsity))
            # get thresholds per channel axis.
            thresholds, _ = torch.kthvalue(
                x.transpose(1, 2).reshape(N * L, C), k=to_drop, dim=0
            )

            # update thresholds using momentum
            self.thresholds.data = (
                self.thresholds.data * (1 - self.momentum) + thresholds * self.momentum
            )

        # subtract off the thresholds and apply relu
        x = x - self.thresholds[None, :, None]
        if hasattr(self, "relu_module"):
            return self.relu_module(x)
        else:
            return torch.nn.functional.relu(x)

    def thresholds_numpy(self, num_motifs):
        # get thresholds per channel axis.
        assert num_motifs == len(self.thresholds)
        return self.thresholds.detach().cpu().numpy()

    def motif_index(self, num_channels):
        # cannot have dropped motifs
        return
