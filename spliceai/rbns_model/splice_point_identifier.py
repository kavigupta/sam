import torch
import torch.nn as nn

from spliceai_torch import ResidualUnit


class SplicePointIdentifier(nn.Module):
    def __init__(self, cl, asymmetric_cl, hidden_size, n_layers=3, starting_channels=4):
        super().__init__()
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

    def forward(self, x, collect_intermediates=False, use_as_motif=True):
        for layer in self.conv_layers:
            x = layer(x)
            x = self.activation(x)
        x = self.last_layer(x)
        if collect_intermediates:
            return dict(output=x)
        return x


class ResidualSplicePointIdentifier(nn.Module):
    def __init__(self, cl, asymmetric_cl, hidden_size, n_layers=3, starting_channels=4):
        super().__init__()
        assert cl % 2 == 0
        self.first_layer = AsymmetricConv(
            starting_channels, hidden_size, cl, *asymmetric_cl
        )
        assert n_layers % 2 == 0
        self.conv = nn.Sequential(
            *[ResidualUnit(hidden_size, w=1, ar=1) for _ in range(n_layers // 2)]
        )
        self.last_layer = nn.Conv1d(hidden_size, 3, 1)

    def forward(self, x, collect_intermediates=False):
        x = x.transpose(2, 1)
        x = self.first_layer(x)
        x = self.conv(x)
        x = self.last_layer(x)
        x = x.transpose(2, 1)
        if collect_intermediates:
            return dict(output=x)
        return x


class AsymmetricConv(nn.Module):
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
