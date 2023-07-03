from abc import ABC, abstractproperty
from torch import nn
import torch
from torch.nn import functional as F

from modular_splicing.utils.construct import construct


class GenericLinearSRMP(nn.Module, ABC):
    """
    Short range motif reprocessor that uses a linear function to
    compute the influence of each motif on the output.

    The linear function is used as a convolution.

    You need to implement the effect_tensor property.
    """

    def __init__(self, radius):
        super().__init__()
        self.radius = radius

    def forward(self, m):
        return m * self.compute_influences(m)

    def compute_influences(self, m):
        m = m.transpose(1, 2)
        m = F.conv1d(m, self.effect_tensor, padding=self.radius)
        m = m.sigmoid()
        m = m.transpose(1, 2)
        return m

    @abstractproperty
    def effect_tensor(self):
        """
        Returns a tensor of shape (num_motifs, num_motifs, 2 * radius + 1) that
            represents the reprocessing effect of each motif on each other motif.
        """
        pass


class LinearSRMP(GenericLinearSRMP):
    """
    Basic linear short range motif reprocessor.
    """

    def __init__(self, radius, num_motifs):
        super().__init__(radius)
        self._effect_tensor = nn.Parameter(
            torch.randn(num_motifs, num_motifs, 2 * radius + 1)
        )

    @property
    def effect_tensor(self):
        return self._effect_tensor


class LinearBatchNormSRMP(LinearSRMP):
    """
    Like a linear short range motif reprocessor, but uses batch normalization
    to normalize the motifs before computing the influence.
    """

    def __init__(self, radius, num_motifs):
        super().__init__(radius, num_motifs)
        self.batch_norm = nn.BatchNorm1d(num_motifs)

    def compute_influences(self, m):
        m = m.transpose(1, 2)
        m = self.batch_norm(m)
        m = m.transpose(1, 2)
        return super().compute_influences(m)


class LinearPostBatchNormSRMP(LinearSRMP):
    """
    Like a linear short range motif reprocessor, but uses batch normalization
    to normalize the motifs after the convolutional layer but before multiplying
    through.

    This means the influences are not clamped to (0, 1) anymore.
    """

    def __init__(self, radius, num_motifs):
        super().__init__(radius, num_motifs)
        self.batch_norm = nn.BatchNorm1d(num_motifs)

    def compute_influences(self, m):
        m = super().compute_influences(m)
        m = m.transpose(1, 2)
        m = self.batch_norm(m)
        m = m.transpose(1, 2)
        return m


class ResidualStackSRMP(nn.Module):
    """
    Represents a generic influence propagator, that consists of
    running a convolutional layer on the influence tensor and then
    combining that with the original in some way.

    Arguments
    ---------
    num_motifs: int
        The number of motifs.
    radius: int
        The radius of the reprocessing layer's context window.
    channels: int
        The number of channels in the reprocessing layer.
    depth: int
        The number of ResidualUnits in the reprocessing layer.
    influence_propagation_spec: dict
        A dictionary specifying the influence propagation layer.
    """

    def __init__(
        self,
        num_motifs,
        radius,
        channels,
        depth,
        influence_propagation_spec=dict(type="MultiplyBySigmoid"),
    ):
        super().__init__()

        from modular_splicing.models.modules.residual_unit_stack import ResidualStack

        self.stack = ResidualStack(
            input_channels=num_motifs,
            hidden_channels=channels,
            output_channels=num_motifs,
            width=2 * radius + 1,
            depth=depth,
        )

        self.influence_propagation = construct(
            influence_propagation_types(), influence_propagation_spec
        )

    def forward(self, m):
        influences = self.stack(m)
        propagation = getattr(self, "influence_propagation", MultiplyBySigmoid())
        return propagation(m, influences)


def srmp_types():
    return dict(
        LinearSRMP=LinearSRMP,
        ResidualStackSRMP=ResidualStackSRMP,
        LinearBatchNormSRMP=LinearBatchNormSRMP,
        LinearPostBatchNormSRMP=LinearPostBatchNormSRMP,
    )


def influence_propagation_types():
    return dict(
        MultiplyBySigmoid=MultiplyBySigmoid,
        PropagateSparsityAndSum=PropagateSparsityAndSum,
    )


class MultiplyBySigmoid(nn.Module):
    """
    Multiply the motif matrix by the influence tensor, but sigmoided.
    """

    def forward(self, m, influences):
        return m * influences.sigmoid()


class PropagateSparsityAndSum(nn.Module):
    """
    Add the motif matrix to the influence tensor, but only where the motif matrix
    is non-zero.
    """

    def forward(self, m, influences):
        return (m + influences) * (m != 0).float()
