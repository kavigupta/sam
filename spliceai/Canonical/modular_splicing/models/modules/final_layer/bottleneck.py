import torch
import torch.nn as nn


class ConvolutionalBottleneck(nn.Module):
    def __init__(self, *, num_motifs, num_output_motifs=2, width):
        super().__init__()
        assert width % 2 == 1
        self.num_motifs = num_motifs
        self.conv = nn.Conv1d(num_motifs, num_output_motifs, width, padding=width // 2)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        padding = torch.zeros(
            x.shape[0], self.num_motifs - x.shape[1], x.shape[2], device=x.device
        )
        x = torch.cat(
            [x, padding],
            dim=1,
        )
        x = x.transpose(1, 2)
        return x


class FirstChannelsBottleneck(nn.Module):
    def __init__(self, *, num_motifs, num_outputs_kept):
        super().__init__()
        self.num_motifs = num_motifs
        self.num_outputs_kept = num_outputs_kept

    def forward(self, x):
        assert len(x.shape) == 3
        assert x.shape[2] == self.num_motifs, str((x.shape, self.num_motifs))
        x = x[:, :, : self.num_outputs_kept]
        padding = torch.zeros(
            x.shape[0],
            x.shape[1],
            self.num_motifs - x.shape[2],
            device=x.device,
        )
        x = torch.cat([x, padding], dim=2)
        return x


def bottleneck_types():
    return dict(
        Identity=lambda num_motifs: nn.Identity(),
        ConvolutionalBottleneck=ConvolutionalBottleneck,
        FirstChannelsBottleneck=FirstChannelsBottleneck,
    )
