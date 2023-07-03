import torch
import torch.nn as nn

from .linear_effects import LinearEffects


class FullTableLinearEffects(LinearEffects):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mu = nn.Parameter(
            torch.randn(self.num_motifs, self.num_motifs, self.width)
        )

    def compute_mu(self):
        return self.mu
