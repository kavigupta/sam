import torch.nn as nn
from modular_splicing.models.modules.influence_value.linear_effects.linear_effects import (
    linear_effects_types,
)

from modular_splicing.utils.construct import construct

from modular_splicing.models.modules.residual_unit_stack import ResidualStack
from .single_attention_long_range import SingleAttentionLongRangeProcessor
from .tanh_long_range import (
    MultiRoundMotifInfluenceTanhLongRangeProcessor,
    SingleTanhLongRangeProcessor,
)


class InfluenceValueCalculator(nn.Module):
    """
    Computes the influence value of each motif in the input sequence.

    First applies a sparse reprocessor to the sequence, then applies a long range
    reprocessor to the sequence. The long range reprocessor uses attention to compute
    the influence value of each motif in the sequence.

    Shape: (batch_size, seq_len, num_motifs) -> (batch_size, seq_len, num_motifs)
    """

    def __init__(
        self,
        *,
        num_motifs,
        channels,
        cl,
        post_sparse_spec,
        long_range_reprocessor_spec,
        intermediate_channels=None,
        selector_spec=dict(type="Identity"),
        activation_spec=dict(type="Identity"),
    ):
        super().__init__()

        if intermediate_channels is None:
            intermediate_channels = num_motifs

        self.sparse_reprocessor = construct(
            post_sparse_types(),
            post_sparse_spec,
            input_channels=num_motifs,
            hidden_channels=channels,
            output_channels=intermediate_channels,
        )

        self.long_range_reprocessor = construct(
            long_range_reprocessor_types(),
            long_range_reprocessor_spec,
            num_motifs=intermediate_channels,
            cl=cl,
        )

        self.selector = construct(
            self.selector_types(),
            selector_spec,
            in_c=intermediate_channels,
            out_c=num_motifs,
        )
        self.activation = construct(dict(Identity=nn.Identity), activation_spec)

    def forward(self, output, collect_intermediates):
        output = self.sparse_reprocessor(output)
        attn_output, output = self.long_range_reprocessor(output, collect_intermediates)
        output = self.selector(output)
        output = self.activation(output)
        return attn_output, output

    @staticmethod
    def selector_types():
        def identity(in_c, out_c):
            assert in_c == out_c
            return nn.Identity()

        def linear(in_c, out_c):
            return nn.Linear(in_c, out_c)

        return dict(Identity=identity, Linear=linear)


def influence_value_calculator_types():
    return dict(
        InfluenceValueCalculator=InfluenceValueCalculator, **linear_effects_types()
    )


def post_sparse_types():
    from .post_sparse import (
        ReducedDimensionalityPostSparse,
        SparsityPropagationPostSparse,
        LinearConvPostSparse,
    )

    return dict(
        ResidualStack=ResidualStack,
        Linear=lambda input_channels, hidden_channels, output_channels: nn.Linear(
            input_channels, output_channels
        ),
        Identity=lambda input_channels, hidden_channels, output_channels: nn.Identity(),
        ReducedDimensionalityPostSparse=ReducedDimensionalityPostSparse,
        SparsityPropagationPostSparse=SparsityPropagationPostSparse,
        LinearConvPostSparse=LinearConvPostSparse,
    )


def long_range_reprocessor_types():
    return dict(
        SingleAttentionLongRangeProcessor=SingleAttentionLongRangeProcessor,
        SingleTanhLongRangeProcessor=SingleTanhLongRangeProcessor,
        MultiRoundMotifInfluenceTanhLongRangeProcessor=MultiRoundMotifInfluenceTanhLongRangeProcessor,
    )
