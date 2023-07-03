import torch
import torch.nn as nn

from .linear_effects import LinearEffects


class LinearlyLatentLinearEffects(LinearEffects):
    def __init__(self, *, latent_dimension, **kwargs):
        super().__init__(**kwargs)
        self.mu_in_latent_space = nn.Parameter(
            torch.randn(self.num_motifs, self.num_motifs, latent_dimension)
        )
        self.latent_space_vectors = nn.Parameter(
            torch.randn(latent_dimension, self.width)
        )

    def compute_mu(self):
        return self.mu_in_latent_space @ self.latent_space_vectors
