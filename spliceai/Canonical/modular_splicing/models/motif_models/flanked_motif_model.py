import torch
import torch.nn as nn

from modular_splicing.utils.construct import construct

from modular_splicing.models.motif_models.types import motif_model_types


class FlankedMotifModel(nn.Module):
    """
    Represents a model that augments a motif model with a `flanking model`, which
        is effectively a single PSAM that is applied to a wider region, for each motif.

    We also have the option of weighting the flanking strength by position, this is
        a single filter trained across all motifs.

    Parameters
    ----------
    model_to_augment_spec : dict
        The specification for the model to augment.
    flanking_size : int
        The size of the flanking region to apply the flanking model to.
    num_motifs : int
        The number of motifs to use.
    flanking_strength_position_specific : bool
        Whether to use a position specific flanking strength.
    """

    def __init__(
        self,
        *,
        model_to_augment_spec,
        flanking_size,
        num_motifs,
        flanking_strength_position_specific=False,
        **kwargs,
    ):
        super().__init__()
        self.model_to_augment = construct(
            motif_model_types(), model_to_augment_spec, num_motifs=num_motifs, **kwargs
        )
        self.flanking_filter = nn.Conv1d(
            in_channels=4, out_channels=num_motifs, kernel_size=1
        )
        self.flanking_size = flanking_size
        if flanking_strength_position_specific:
            self.flanking_strength_position_specific = nn.Parameter(
                torch.zeros(1, 1, self.flanking_size)
            )
        else:
            self.flanking_strength_position_specific = None

    def single_flank(self, x):
        x = x.transpose(1, 2)
        x = self.flanking_filter(x)
        x = x.transpose(1, 2)
        return x

    def flanking_weight(self, device):
        logits = getattr(self, "flanking_strength_position_specific", None)
        if logits is not None:
            return logits.sigmoid()
        return torch.ones(1, 1, self.flanking_size).to(device)

    def smear_flank(self, flank):
        # N, L, C
        flank = flank.transpose(1, 2)
        # N, C, L
        original_shape = flank.shape
        flank = flank.reshape(flank.shape[0] * flank.shape[1], flank.shape[2])
        # N * C, L
        flank = flank.unsqueeze(1)
        # N * C, 1, L
        weight = self.flanking_weight(flank.device)
        flank = nn.functional.conv1d(flank, weight, padding=self.flanking_size // 2)
        flank = flank.squeeze(1)
        # N * C, L
        flank = flank.reshape(original_shape)
        # N, C, L
        flank = flank.transpose(1, 2)
        # N, L, C
        return flank

    def forward(self, x):
        if isinstance(x, dict):
            x = x["x"]
        motifs = self.model_to_augment(x)
        flank = self.single_flank(x)
        flank = self.smear_flank(flank)
        return flank + motifs

    def notify_sparsity(self, sparsity):
        self.model_to_augment.notify_sparsity(sparsity)
