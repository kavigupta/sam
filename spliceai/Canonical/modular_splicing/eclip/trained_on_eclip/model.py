from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from modular_splicing.models.modules.sparsity_enforcer.sparsity_enforcer import (
    sparsity_enforcer_types,
)
from modular_splicing.models.motif_models.types import motif_model_types
from modular_splicing.utils.construct import construct


class EclipMatchingModel(nn.Module, ABC):
    """
    Generic class for an eCLIP matching model

    Predicts the probability that a given sequence is not a match to a given motif,
        by taking the assumption of an independent AND relationship between each position

    The `forward` method of this module predicts the log probability that every given position
        is not a match to the given motif
    """

    @abstractmethod
    def forward(self, x):
        pass

    def summary(self, x):
        """
        Predicts log P(not a match) overall
        """
        return self(x).sum(-1)

    def predict(self, x):
        """
        Produce the probability of an entire sequence *being* a match.

        Note: not in log space.
        """
        return 1 - self.summary(x).exp()

    def loss(self, x, y):
        """
        Computes the loss for a given batch of sequences and their labels.

        Labels should be 0 for not a match, 1 for a match.

        We complement the labels, because `summary` is predicting the probability of *not* being a match.
        """
        return nn.BCEWithLogitsLoss()(self.summary(x), 1 - y)


class EclipMatchingModelAM(EclipMatchingModel):
    """
    AM based eclip matching model

    Parameters
    ----------
    k : int
        The wiggle room in the AM motif sparsity
    w : int
        The width of the AM motif
    motif : str
        The motif to use, e.g., "TRA2A"
    psam_source : str
        The source of the psams, e.g., "rbns".
    sparsity : float
        The density to enforce in the AM motif
    post_sparse_scale : bool
        Whether to scale the output of the AM motif by a learnable parameter
        This can help with ensuring that our model is well-conditioned.
    """

    def __init__(self, k, w, motif, psam_source, sparsity, post_sparse_scale=False):
        super().__init__()

        self.motif_model = construct(
            motif_model_types(),
            self.motif_model_spec(k=k, w=w, motif=motif, psam_source=psam_source),
            input_size=4,
            channels=200,
            num_motifs=1,
        )
        self.sparse_layer = construct(
            sparsity_enforcer_types(),
            dict(
                type="SparsityEnforcer",
                num_motifs=1,
                sparse_spec=dict(type="SpatiallySparseAcrossChannels"),
                sparsity=sparsity,
            ),
        )
        self.motif_model.notify_sparsity(self.sparse_layer.sparse_layer.get_sparsity())
        if post_sparse_scale:
            self.post_sparse_scale = nn.Parameter(torch.randn(1))
        else:
            self.post_sparse_scale = None

    def forward(self, x):
        x = self.motif_model(x)["motifs"]
        _, x = self.sparse_layer.forward(
            x,
            splicepoint_results=None,
            manipulate_post_sparse=None,
            collect_intermediates=False,
        )
        x = x.squeeze(-1)
        post_sparse_scale = getattr(self, "post_sparse_scale", None)
        if post_sparse_scale is not None:
            x = x * post_sparse_scale.abs()
        return -x

    @staticmethod
    def motif_model_spec(*, k, w, motif, psam_source):
        assert w % 2 == 1
        depth = (w - 1) / 4
        if depth == int(depth):
            depth = int(depth)
        return dict(
            type="AdjustedMotifModel",
            model_to_adjust_spec=dict(
                type="PSAMMotifModel",
                motif_spec={"type": psam_source},
                include_names=[motif],
            ),
            adjustment_model_spec=dict(
                type="LearnedMotifModel",
                motif_width=w,
                motif_fc_layers=5,
                motif_feature_extractor_spec=dict(type="ResidualStack", depth=depth),
            ),
            sparsity_enforcer_spec=dict(
                type="SparsityEnforcer",
                sparse_spec=dict(type="SpatiallySparseAcrossChannels"),
            ),
            sparsity_multiplier=k,
        )
