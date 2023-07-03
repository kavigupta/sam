from abc import ABC, abstractmethod
import torch.nn as nn
import torch.nn.functional as F


class LinearEffects(nn.Module, ABC):
    """
    Represents how to compute true motifs when effects are linear.

    In effect, we set up the optimization problem (where m0 is the input and m is the true
        motifs we output):

        min_m ||m - m0||^2 - m^T Lambda m

    The equation of solutions here is

        m - m0 = Lambda m

    We can thus solve this iteratively by

        m_{t+1} = Lambda m_t + m0

    Which can be interpreted as the original motifs + the influenced motifs.

    We choose to enforce Lambda to be a convolution. This is because we want to enforce
        that the influence of a motif is only on the motifs that are close to it, and
        that the influence is translationally invariant.

    We thus produce a parameter `mu`, which is of shape (num_motifs, num_motifs, width)
    """

    def __init__(
        self, num_motifs, cl, num_iterations, channels, enforce_stability=False
    ):
        super().__init__()
        del channels
        self.num_motifs = num_motifs
        self.width = cl + 1
        assert self.width % 2 == 1
        self.num_iterations = num_iterations
        self.enforce_stability = enforce_stability

    @abstractmethod
    def compute_mu(self):
        """
        Produce the mu parameter, which is of shape (num_motifs, num_motifs, width)
        """
        pass

    def forward(self, m0, collect_intermediates):
        """
        Compute the true motifs given the input motifs m0

        :param m0: The input motifs, of shape (batch_size, num_motifs, width)
        :return: The true motifs, of shape (batch_size, num_motifs, width)
        """
        intermediates = {}
        mu = self.compute_mu()
        if collect_intermediates:
            intermediates["linear_effects_mu"] = mu
        m0 = m0.transpose(1, 2)
        target_mean = m0.mean((1, 2))
        if collect_intermediates:
            intermediates["processed_motifs_0"] = m0
        m = m0
        for i in range(self.num_iterations):
            m = F.conv1d(m, mu, padding=self.width // 2) + m0
            m = F.relu(m)
            if getattr(self, "enforce_stability", False):
                adjustment_factor = target_mean / (1e-5 + m.mean((1, 2)))
                m = m * adjustment_factor[:, None, None]
            if collect_intermediates:
                intermediates[f"processed_motifs_{i + 1}"] = m
        m = m.transpose(1, 2)
        return intermediates, m


def linear_effects_types():
    from .full_table_linear_effects import FullTableLinearEffects
    from .linearly_latent_linear_effects import LinearlyLatentLinearEffects
    from .sum_gaussian_linear_effects import SumGaussianLinearEffects

    return dict(
        FullTableLinearEffects=FullTableLinearEffects,
        LinearlyLatentLinearEffects=LinearlyLatentLinearEffects,
        SumGaussianLinearEffects=SumGaussianLinearEffects,
    )
