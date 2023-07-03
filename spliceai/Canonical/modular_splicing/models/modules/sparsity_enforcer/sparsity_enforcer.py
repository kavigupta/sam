import torch
import torch.nn as nn

from modular_splicing.utils.construct import construct

from modular_splicing.models.modules.sparsity.sparsity import sparse_layer_types

from .presparse_normalizer import (
    BasicPresparseNormalizer,
    NoPresparseNormalizer,
)


class SparsityEnforcer(nn.Module):
    """
    Enforces sparsity on the motifs.

    First, runs them through a presparse normalizer, which is a batch normalization layer, then
    runs them through a sparse layer, which enforces sparsity on the motifs.

    Parameters
    ----------
    num_motifs : int
        The number of motifs.
    sparse_spec : dict
        The specification for the sparse layer.
    sparsity : float
        The sparsity to enforce.
    affine : bool
        Whether to use affine batch normalization. If False, the inputs to the sparse layer will
        all be a standard normal (mean 0, variance 1). Otherwise, the inputs will have a trainable
        distribution and thus the sparse layer (if set appropriately) can choose which motifs to
        prioritize for greater sparsity.
    """

    def __init__(
        self,
        num_motifs,
        sparse_spec,
        sparsity,
        affine=False,
        presparse_normalizer_spec=dict(type="BasicPresparseNormalizer"),
    ):
        super().__init__()
        self.presparse_norm = construct(
            dict(
                NoPresparseNormalizer=NoPresparseNormalizer,
                BasicPresparseNormalizer=BasicPresparseNormalizer,
            ),
            presparse_normalizer_spec,
            num_motifs=num_motifs,
            affine=affine,
        )

        self.sparse_layer = construct(
            sparse_layer_types(),
            sparse_spec,
            sparsity=1 - sparsity,
            num_channels=num_motifs,
        )

    def forward(
        self, x, splicepoint_results, manipulate_post_sparse, collect_intermediates
    ):
        full_outputs = {}

        def add(**kwargs):
            if collect_intermediates:
                full_outputs.update(kwargs)

        x = x.transpose(1, 2)
        x = self.presparse_norm(x)
        add(nonsparse_motifs=x.transpose(1, 2))
        x = self.sparse_layer(x)
        x = x.transpose(1, 2)
        if manipulate_post_sparse is not None:
            x = manipulate_post_sparse(x)

        add(post_sparse_motifs_only=x)

        if splicepoint_results is not None:
            x = torch.cat([splicepoint_results, x], dim=2)

        add(post_sparse=x)
        return full_outputs, x

    @property
    def num_motifs(self):
        return self.presparse_norm.num_features

    @property
    def thresholds_numpy(self):
        return self.sparse_layer.thresholds_numpy(self.num_motifs)

    def motif_index(self):
        """
        Like SparseLayer.motif_index, but ensures that the result is
        a dictionary. By default just sends every motif to itself.
        """
        motif_index = self.sparse_layer.motif_index(self.num_motifs)
        if motif_index is not None:
            return motif_index
        return {k: k for k in range(self.num_motifs)}


def sparsity_enforcer_types():
    from shelved.robustly_adjusted.multi_motif_parallel_sparsity_enforcer import (
        MultiMotifParallelSparsityEnforcer,
    )

    return dict(
        SparsityEnforcer=SparsityEnforcer,
        MultiMotifParallelSparsityEnforcer=MultiMotifParallelSparsityEnforcer,
    )
