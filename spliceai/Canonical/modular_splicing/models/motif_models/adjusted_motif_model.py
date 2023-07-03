import torch.nn as nn
from modular_splicing.models.modules.sparsity_enforcer.sparsity_enforcer import (
    sparsity_enforcer_types,
)
from modular_splicing.models.motif_models.types import motif_model_types

from modular_splicing.utils.construct import construct


class AdjustedMotifModel(nn.Module):
    """
    Adjusted motif model.

    This is a motif model that adjusts the output of another motif model by
    adding the output of an adjustment model. The adjustment model is
    another motif model.

    Arguments
    ---------
    model_to_adjust_spec : spec
        The spec for the model to adjust.
    adjustment_model_spec : spec
        The spec for the adjustment model.
    sparsity_enforcer_spec : spec
        The spec for the sparsity enforcer to use on the model to adjust.
    sparsity_multiplier : float
        The multiplier to use on the sparsity of the model to adjust.
    """

    _input_is_dictionary = True
    motif_model_dict = True

    def __init__(
        self,
        model_to_adjust_spec,
        adjustment_model_spec,
        sparsity_enforcer_spec,
        sparsity_multiplier=2,
        **kwargs,
    ):
        super().__init__()
        self.model_to_adjust = construct(
            motif_model_types(), model_to_adjust_spec, **kwargs
        )
        self.adjustment_model = construct(
            motif_model_types(), adjustment_model_spec, **kwargs
        )
        self.model_to_adjust_sparse = construct(
            sparsity_enforcer_types(),
            sparsity_enforcer_spec,
            num_motifs=kwargs["num_motifs"],
            sparsity=0.5,
        )
        self.sparsity_multiplier = sparsity_multiplier

    def forward(self, x):
        to_adjust = self.model_to_adjust(x)
        if getattr(self.model_to_adjust, "motif_model_dict", False):
            to_adjust = to_adjust["motifs"]
        # sparsify
        _, to_adjust = self.model_to_adjust_sparse(
            to_adjust,
            splicepoint_results=None,
            manipulate_post_sparse=None,
            collect_intermediates=False,
        )
        # create adjustment value
        adjustment = self.adjustment_model(x)
        if getattr(self.adjustment_model, "motif_model_dict", False):
            adjustment = adjustment["motifs"]
        # adjust
        adjustment = adjustment * (to_adjust != 0).float()
        return dict(motifs=to_adjust + adjustment, pre_adjustment_motifs=to_adjust)

    def notify_sparsity(self, sparsity):
        # set the density of the model to `sparsity_multiplier` times the
        # provided density.
        less_sparse = max(
            1 - (1 - sparsity) * getattr(self, "sparsity_multiplier", 2), 0
        )
        print(f"Updating adjusted sparsity to {less_sparse}")
        self.model_to_adjust_sparse.sparse_layer.set_sparsity(less_sparse)
