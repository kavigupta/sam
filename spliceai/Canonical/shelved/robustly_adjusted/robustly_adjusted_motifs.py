import numpy as np
import torch
from torch import nn

from modular_splicing.utils.construct import construct

from modular_splicing.models.motif_models.types import motif_model_types


class RobustlyAdjustedMotifs(nn.Module):
    motif_model_dict = True

    def __init__(self, adjusted_model_spec, randomize_after_sparse=False, **kwargs):
        super().__init__()
        self.adjusted_model = construct(
            motif_model_types(), adjusted_model_spec, **kwargs
        )
        self.rng = np.random.RandomState(0)
        self.randomize_after_sparse = randomize_after_sparse

    def forward(self, x):
        result = self.adjusted_model(x)
        if self.training:
            self.process(result)
        return result

    def process(self, result):
        if getattr(self, "randomize_after_sparse", False):
            result["sparsity_enforcer_extra_params"] = dict(
                choice_indices=self.choice_mask_for(result["motifs"]).long(),
                other_inputs=[result["pre_adjustment_motifs"]],
            )
        else:
            result["motifs"] = self.randomly_interleave(
                result["motifs"], result["pre_adjustment_motifs"]
            )

    def notify_sparsity(self, sparsity):
        self.adjusted_model.notify_sparsity(sparsity)

    def randomly_interleave(self, x, y):
        choice_fn = self.choice_mask_for(x)
        return torch.where(choice_fn, x, y)

    def choice_mask_for(self, x):
        return torch.tensor(self.rng.choice(2, size=x.shape).astype(np.bool)).to(
            x.device
        )
