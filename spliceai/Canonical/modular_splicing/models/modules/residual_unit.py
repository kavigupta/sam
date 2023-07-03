import torch.nn as nn

from modular_splicing.utils.construct import construct
from .activations import activation_types


class ResidualUnit(nn.Module):
    """
    Residual unit proposed in "Identity mappings in Deep Residual Networks"
    by He et al.

    Shapes: (N, C, L) -> (N, C, L)

    True width is 2 * w - 1, when ar = 1.
    """

    def __init__(self, l, w, ar, *, nonlinearity_spec=dict(type="ReLU")):
        super().__init__()
        self.normalize1 = nn.BatchNorm1d(l)
        self.normalize2 = nn.BatchNorm1d(l)
        self.act1 = self.act2 = construct(activation_types(), nonlinearity_spec)

        padding = (ar * (w - 1)) // 2

        self.conv1 = nn.Conv1d(l, l, w, dilation=ar, padding=padding)
        self.conv2 = nn.Conv1d(l, l, w, dilation=ar, padding=padding)

    def forward(self, input_node):
        bn1 = self.normalize1(input_node)
        act1 = self.act1(bn1)
        conv1 = self.conv1(act1)
        assert conv1.shape == act1.shape
        bn2 = self.normalize2(conv1)
        act2 = self.act2(bn2)
        conv2 = self.conv2(act2)
        assert conv2.shape == act2.shape
        output_node = conv2 + input_node
        return output_node


class HalfResidualUnit(nn.Module):
    """
    Similar to the ResidualUnit above but only half of it.

    Shapes: (N, C, L) -> (N, C, L)

    True width is w, when ar = 1.
    """

    def __init__(self, l, w, ar):
        super().__init__()
        self.normalize1 = nn.BatchNorm1d(l)
        self.act1 = nn.ReLU()

        padding = (ar * (w - 1)) // 2

        self.conv1 = nn.Conv1d(l, l, w, dilation=ar, padding=padding)

    def forward(self, input_node):
        bn1 = self.normalize1(input_node)
        act1 = self.act1(bn1)
        conv1 = self.conv1(act1)
        assert conv1.shape == act1.shape
        output_node = conv1 + input_node
        return output_node
