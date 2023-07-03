import torch
import torch.nn as nn
from modular_splicing.models.motif_models.types import motif_model_types

from modular_splicing.utils.construct import construct


class ParallelMotifModels(nn.Module):
    """
    Run several motif models in parallel, and concatenate the results.

    Arguments
    ---------
    specs: list of specs
        The specs for each motif model
    num_motifs_each: list of ints
        The number of motifs for each motif model
    num_motifs: int
        The total number of motifs. Must be the sum of num_motifs_each.
    notify_sparsity_idxs: list of ints
        The indices of the motif models to notify of sparsity. If None, notify
        all of them.
    """

    _input_is_dictionary = True

    def __init__(
        self, *, specs, num_motifs_each, num_motifs, notify_sparsity_idxs=None, **kwargs
    ):
        super().__init__()
        assert sum(num_motifs_each) == num_motifs
        self.parallel_models = nn.ModuleList(
            [
                construct(motif_model_types(), spec, num_motifs=nm, **kwargs)
                for spec, nm in zip(specs, num_motifs_each)
            ]
        )
        self.notify_sparsity_idxs = notify_sparsity_idxs

    def forward(self, x):
        results = []
        for m in self.parallel_models:
            motifs = m(x if getattr(m, "_input_is_dictionary", False) else x["x"])
            if getattr(m, "motif_model_dict", False):
                motifs = motifs["motifs"]
            results.append(motifs)
        return torch.cat(results, dim=2)

    def notify_sparsity(self, sparsity):
        notify_sparsity_idxs = getattr(self, "notify_sparsity_idxs", None)
        if notify_sparsity_idxs is None:
            notify_sparsity_idxs = list(range(len(self.parallel_models)))
        for idx in notify_sparsity_idxs:
            self.parallel_models[idx].notify_sparsity(sparsity)
