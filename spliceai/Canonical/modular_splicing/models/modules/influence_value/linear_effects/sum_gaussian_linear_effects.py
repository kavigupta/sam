import torch
import torch.nn as nn

from .linear_effects import LinearEffects


class SumGaussianLinearEffects(LinearEffects):
    def __init__(self, *, num_gaussians_each, **kwargs):
        """
        For each 79 x 79 table of motif effects, we have num_gaussians_each gaussians,
            each with a mean and a standard deviation, and scalar
        """
        super().__init__(**kwargs)
        self.num_gaussians_each = num_gaussians_each
        self.means, self.stds, self.scalars = [
            nn.Parameter(
                torch.randn(self.num_motifs, self.num_motifs, num_gaussians_each, 1)
            )
            for _ in range(3)
        ]

    def compute_mu(self):
        t = torch.linspace(-1, 1, self.width, device=self.means.device)
        # t : (W, )
        t = t[None, None, None, :]
        # t : (1, 1, 1, W)
        t = ((t - self.means) / self.stds) ** 2
        t = torch.exp(-t / 2)
        # t : (M, M, G, W)
        t = t * self.scalars
        # t : (M, M, G, W)
        t = t.sum(dim=2)
        # t : (M, M, W)
        return t
