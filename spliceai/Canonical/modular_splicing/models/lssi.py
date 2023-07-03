import torch
import torch.nn as nn


class SplicePointIdentifier(nn.Module):
    """
    Represnts the core LSSI model. This is a convolutional neural network that takes in a sequence of RNA
    and outputs a prediction of the splicing probability at each position in the sequence.

    It only represents one of 3' or 5' splice sites, you should train and use two of these models to predict
    both splice sites.

    The model architecture is a first layer, which is an asymmetric convolution covering the asymmetric_cl,
        followed by n_layers layers of 1x1 convolutions, followed by a final 1x1 convolution to produce the
        output.

    Arguments:
        cl: The "context length" of the model. This is the amount of data
            that the model will clip from either side. Exists primarily for
            compatibility with the data code, that assumes an equal context
            length on either side
        asymmetric_cl: The asymmetric context length. This is a tuple of
            (left, right) context lengths. The left context length is the
            amount of data that the model will clip from the left side, and
            the right context length is the amount of data that the model
            will clip from the right side. If this is None, then the model
            will use a symmetric context length equal to cl.
        hidden_size: The number of channels in the hidden layers
        n_layers: The number of hidden layers
        starting_channels: The number of channels in the input
        input_size: The size of the input. This is not used, but is kept
            for compatibility with the training code.
        sparsity: Unused. Kept for compatibility with the training code.
    """

    def __init__(
        self,
        cl,
        asymmetric_cl,
        hidden_size,
        n_layers=3,
        starting_channels=4,
        input_size=4,
        sparsity=None,
    ):
        super().__init__()
        del input_size, sparsity
        assert cl % 2 == 0
        if asymmetric_cl is None:
            first_layer = nn.Conv1d(starting_channels, hidden_size, cl + 1)
        else:
            first_layer = AsymmetricConv(
                starting_channels, hidden_size, cl, *asymmetric_cl
            )
        conv_layers = [first_layer] + [
            nn.Conv1d(hidden_size, hidden_size, 1) for _ in range(n_layers)
        ]
        self.conv_layers = nn.ModuleList(conv_layers)
        self.activation = nn.ReLU()
        self.last_layer = nn.Conv1d(hidden_size, 3, 1)

    def forward(self, x, collect_intermediates=False, collect_losses=False):
        if isinstance(x, dict):
            x = x["x"]
        x = x.transpose(2, 1)
        for layer in self.conv_layers:
            x = layer(x)
            x = self.activation(x)
        x = self.last_layer(x)
        x = x.transpose(2, 1)
        if collect_intermediates or collect_losses:
            return dict(output=x)
        return x


class AsymmetricConv(nn.Module):
    """
    Represents an asymmetric convolution. This is a convolution that has a different context length on the
    left and right sides.

    In practice, this is implemented by padding the input with zeros on the left and right sides, and then
    performing a normal convolution.
    """

    clipping = "cl-based"

    def __init__(self, in_channels, out_channels, cl, left, right):
        super().__init__()
        assert cl % 2 == 0
        assert max(left, right) <= cl // 2
        self.conv = nn.Conv1d(in_channels, out_channels, left + right + 1)
        self.cl = cl
        self.left = left
        self.right = right

    def forward(self, x):
        x = torch.cat(
            [
                torch.zeros(*x.shape[:-1], self.left).to(x.device),
                x,
                torch.zeros(*x.shape[:-1], self.right).to(x.device),
            ],
            dim=-1,
        )
        x = self.conv(x)
        if self.clipping == "cl-based":
            x = x[:, :, self.cl // 2 : x.shape[-1] - self.cl // 2]
        elif self.clipping == "natural":
            x = x[:, :, self.left : -self.right]
        elif self.clipping == "none":
            pass
        else:
            raise RuntimeError(f"bad value for self.clipping: {self.clipping}")
        return x
