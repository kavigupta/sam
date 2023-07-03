import torch.nn as nn

from modular_splicing.models.modules.residual_unit import ResidualUnit


class FCMotifFeatureExtractor(nn.Module):
    """
    Module that extracts features from a sequence of motifs.

    First it projects the input to a higher dimension, then it
    performs a number of 1x1 convolutions, then it projects back
    to the output channels.

    Shape: (N, input_channels, L) -> (N, hidden_channels, L)

    Parameters
    ----------
    input_channels : int
        Number of input channels.
    extra_compute : int
        Multiplier for the number of compute channels.
    hidden_channels : int
        Number of hidden channels.
    width : int
        Width of the initial convolution.
    num_compute_layers : int
        Number of 1x1 convolutions.
    """

    def __init__(
        self,
        input_channels,
        extra_compute,
        hidden_channels,
        width,
        num_compute_layers,
    ):
        super().__init__()
        compute_channels = hidden_channels * extra_compute
        output_channels = hidden_channels
        assert width % 2 == 1
        self.proj_in = nn.Conv1d(
            input_channels,
            compute_channels,
            kernel_size=width,
            padding=width // 2,
        )
        self.compute = get_fc_block(num_compute_layers, compute_channels)
        self.attn = nn.ReLU()
        self.proj_out = nn.Conv1d(compute_channels, output_channels, kernel_size=1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.proj_in(x)
        x = self.compute(x)
        x = self.attn(x)
        x = self.proj_out(x)
        x = x.transpose(1, 2)
        return x


def get_fc_block(layers, channels):
    """
    Returns a block of `layers` 1x1 convolutions with `channels` channels.

    Parameters
    ----------
    layers : int
        Number of layers.
    channels : int
        Number of channels.
    """
    return nn.Sequential(*[ResidualUnit(l=channels, w=1, ar=1) for _ in range(layers)])
