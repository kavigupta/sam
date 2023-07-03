import torch
import torch.nn as nn


from modular_splicing.utils.construct import construct

from modular_splicing.models.modules.sparsity_enforcer.sparsity_enforcer import (
    sparsity_enforcer_types,
)


class MultiMotifParallelSparsityEnforcer(nn.Module):
    def __init__(self, *, sparsity_enforcer_spec, count, **kwargs):
        super().__init__()
        self.sparse_enforcers = nn.ModuleList(
            [
                construct(sparsity_enforcer_types(), sparsity_enforcer_spec, **kwargs)
                for _ in range(count)
            ]
        )
        self.count = count

    def forward(
        self,
        x,
        *args,
        choice_indices=None,
        other_inputs=None,
        **kwargs,
    ):
        if choice_indices is None:
            assert other_inputs is None
            return self.sparse_enforcers[0](x, *args, **kwargs)
        assert 1 == self.count - 1 == len(other_inputs)
        all_inputs = [x] + other_inputs
        enforcer_outputs = [
            enforcer(input_, *args, **kwargs)
            for enforcer, input_ in zip(self.sparse_enforcers, all_inputs)
        ]
        result = torch.stack([x for _, x in enforcer_outputs], dim=-1)
        choice_indices = nn.functional.pad(choice_indices, [2, 0])

        batch_idxs, seq_idxs, mot_idxs = torch.meshgrid(
            [torch.arange(i) for i in choice_indices.shape]
        )

        return (
            dict(enforcer_outputs=enforcer_outputs),
            result[batch_idxs, seq_idxs, mot_idxs, choice_indices],
        )

    @property
    def num_motifs(self):
        return self.sparse_enforcers[0].num_motifs

    @property
    def thresholds_numpy(self):
        return self.sparse_enforcers[0].thresholds_numpy(self.num_motifs)

    def motif_index(self):
        return self.sparsity_enforcers[0].motif_index()

    @property
    def sparse_layer(self):
        return MultiMotifParallelSparseLayer(self)


class MultiMotifParallelSparseLayer:
    """
    Thin wrapper around the MultiMotifParallelSparsityEnforcer class to make it work with

    """

    def __init__(self, sparsity_enforcer):
        self.sparsity_enforcer = sparsity_enforcer

    def update_sparsity(self, update_by):
        for enforcer in self.sparsity_enforcer.sparse_enforcers:
            enforcer.sparse_layer.update_sparsity(update_by)

    def set_sparsity(self, sparsity):
        for enforcer in self.sparsity_enforcer.sparse_enforcers:
            enforcer.sparse_layer.set_sparsity(sparsity)

    def get_sparsity(self):
        return self.sparsity_enforcer.sparse_enforcers[0].sparse_layer.get_sparsity()

    def forward(self, x):
        raise NotImplementedError()

    def thresholds_numpy(self, num_motifs):
        raise NotImplementedError()

    def motif_index(self, num_channels):
        raise NotImplementedError()
