import torch
import torch.nn as nn


class MotifModel(nn.Module):
    """
    Represents a motif model that classifies a sequence as either
    matching or not matching a given motif.
    """

    def __init__(self, hidden_size, window_size):
        super().__init__()
        self.conv1 = nn.Conv1d(4, hidden_size, kernel_size=5, padding=((5 - 1) // 2))
        self.norm1 = nn.BatchNorm1d(hidden_size)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size * 2, 1)
        self.norm2 = nn.BatchNorm1d(hidden_size * 2)
        self.conv3 = nn.Conv1d(
            hidden_size * 2, hidden_size, kernel_size=window_size - 4
        )
        self.norm3 = nn.BatchNorm1d(hidden_size)
        # self.linear = nn.Linear(hidden_size, 2)
        self.conv4 = nn.Conv1d(hidden_size, 2, 1)
        self.act = nn.ReLU()

    def forward(self, input, f=None, l=None, use_as_motif=False):
        if hasattr(self, "conv2"):
            if use_as_motif:
                pass
            else:
                input = input.transpose(1, 2)  # BxLxC -> BxCxL

            x = self.conv1(input)
            x = self.norm1(x)
            x = self.act(x)
            # x = self.dropout(x)

            x = self.conv2(x)
            x = self.norm2(x)
            x = self.act(x)

            x = self.conv3(x)
            x = self.norm3(x)
            x = self.act(x)
            # x = self.dropout(x)

            # x = torch.squeeze(x)
            # x = self.linear(x)
            x = self.conv4(x)
            if use_as_motif:
                pass
            else:
                x = x.sum(2)
        else:
            if use_as_motif:
                pass
            else:
                input = input.transpose(1, 2)  # BxLxC -> BxCxL

            x = self.conv1(input)
            x = self.norm1(x)
            x = self.act(x)
            # x = self.dropout(x)
            for i in range((self.window_size - 2) // 2):
                x = self.convstack[i](x)

            x = self.conv_output(x)
            if use_as_motif:
                pass
            else:
                x = x.sum(2)

        return x


def _run_rbns_chenxi_model_without_padding(mod, x, *, do_log_softmax):
    """
    Run the Chenxi neural model on the given input, as a motif.

    Output is the result, as well as instructions on how to pad the output.

    See `run_rbns_chenxi_model` for the padding.
    """
    result = x
    result = result.transpose(1, 2)
    result = mod(result, use_as_motif=True)
    result = result.transpose(1, 2)
    if do_log_softmax:
        result = result.log_softmax(-1)
    result = result[:, :, 1]
    pad_total = x.shape[1] - result.shape[1]
    pad_left, pad_right = pad_total // 2, pad_total - pad_total // 2
    return result, pad_left, pad_right


def run_rbns_chenxi_model(mod, x):
    """
    Run the RBNS Chenxi model on the given input, and apply the necessary padding.
    """
    x, left_pad, right_pad = _run_rbns_chenxi_model_without_padding(
        mod, x, do_log_softmax=False
    )

    x = torch.cat(
        [torch.zeros(x.shape[0], left_pad, *x.shape[2:], device=x.device), x], dim=1
    )
    x = torch.cat(
        [x, torch.zeros(x.shape[0], right_pad, *x.shape[2:], device=x.device)], dim=1
    )
    return x
