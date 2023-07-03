import torch
import torch.nn as nn


class LSTMLongRangeFinalLayer(nn.Module):
    """
    Run an LSTM over the input, and return the output.

    Potentially produces extra channels, which it then clips.

    Shape: (batch_size, seq_len, num_channels) -> (batch_size, seq_len, num_channels)

    Parameters
    ----------
    input_channels: int
        The number of input channels.
    output_channels: int
        The number of output channels.
    layers: int
        The number of LSTM layers to use.
    forward_only: bool
        If True, only use the forward LSTM.
    backwards: bool
        If True, run the LSTM backwards.
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        layers=1,
        forward_only=False,
        backwards=False,
    ):
        num_out_channels = layers * (1 if forward_only else 2)
        self.backwards = backwards
        super().__init__()
        self.lstm = nn.LSTM(
            input_channels,
            (output_channels + num_out_channels - 1) // num_out_channels,
            bidirectional=not forward_only,
            num_layers=layers,
        )

    def forward(self, x):
        assert len(x.shape) == 3

        backwards = getattr(self, "backwards", False)
        if backwards:
            x = torch.flip(x, [1])
        original_num_channels = x.shape[2]
        x = x.transpose(0, 1)
        x, _ = self.lstm(x)
        x = x.transpose(0, 1)
        x = x[:, :, :original_num_channels]
        if backwards:
            x = torch.flip(x, [1])
        return x
