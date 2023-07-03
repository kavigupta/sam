import torch
import torch.nn as nn


class NoMotifModel(nn.Module):
    """
    Like a motif model, but just returns 0s. Useful for padding.
    """

    def __init__(self, *, input_size, channels, num_motifs):
        super().__init__()
        self.num_motifs = num_motifs

    def forward(self, sequence):
        return torch.zeros(
            sequence.shape[0],
            sequence.shape[1],
            self.num_motifs,
            device=sequence.device,
            dtype=sequence.dtype,
        )

    def notify_sparsity(self, sparsity):
        pass
