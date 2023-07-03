import torch.nn as nn
from modular_splicing.models.motif_models.types import motif_model_types

from .short_range_motif_reprocessor import srmp_types
from modular_splicing.utils.construct import construct


class ReprocessedMotifModel(nn.Module):
    """
    Motif model that uses a reprocessor to reprocess the motifs before
    passing them on to the next layer.

    Used after Adjusted Motif models as these are sparsified. In
    other cases, this would be redundant.

    Arguments
    ---------
    motif_model_1_spec : spec
        Specification for the motif model to use.
    reprocessor_spec : spec
        Specification for the reprocessor to use.
    num_motifs : int
        Number of motifs to use.
    """

    motif_model_dict = True

    def __init__(
        self,
        *,
        motif_model_1_spec,
        reprocessor_spec,
        num_motifs,
        **kwargs,
    ):
        super().__init__()
        self.motif_model_1 = construct(
            motif_model_types(), motif_model_1_spec, num_motifs=num_motifs, **kwargs
        )
        self.reprocessor = construct(
            srmp_types(),
            reprocessor_spec,
            num_motifs=num_motifs,
        )

    def forward(self, x):
        x = self.motif_model_1(x)
        x["pre_reprocessor"] = x.pop("motifs")
        x["motifs"] = self.reprocessor(x["pre_reprocessor"])
        return x

    def notify_sparsity(self, sparsity):
        self.motif_model_1.notify_sparsity(sparsity)
