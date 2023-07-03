import torch
import torch.nn as nn


from modular_splicing.models.modules.positional_encoding import PositionalEncoding
from modular_splicing.models.modules.sparsity_propagation import (
    sparsity_propagation_types,
)
from modular_splicing.utils.construct import construct

from .tanh_attention.tanh_convolved_attn import TanhConvolvedAttnOnIndices


class SingleTanhLongRangeProcessor(nn.Module):
    """
    Uses the motifs to influence the outputs.
    """

    def __init__(
        self,
        num_motifs,
        cl,
        num_heads=1,
        forward_only=False,
        num_outputs=2,
        embed_dim=128,
        *,
        max_len=10_000,
        only_on_output_channels=True,
        tanh_attn_layer_spec=dict(type="TanhConvolvedAttnLayer"),
        v_proj_spec=dict(type="Linear"),
        positional_encoding_spec=dict(type="PositionalEncoding", dropout=0),
        reproject_after_pe=False,
    ):
        super().__init__()

        self.num_outputs = num_outputs

        self.attn = TanhConvolvedAttnOnIndices(
            total_dim=num_motifs,
            input_idxs=list(
                range(num_outputs, num_motifs)
                if only_on_output_channels
                else range(num_motifs)
            ),
            output_idxs=list(range(num_motifs)),
            embed_dim=embed_dim,
            num_outputs=num_outputs,
            window=cl,
            num_heads=num_heads,
            forward_only=forward_only,
            tanh_attn_layer_spec=tanh_attn_layer_spec,
            max_len=max_len,
            v_proj_spec=v_proj_spec,
            positional_encoding_spec=positional_encoding_spec,
            reproject_after_pe=reproject_after_pe,
        )

    def forward(self, output, collect_intermediates):
        num_total = output.shape[-1]
        attn_output = self.attn(output, collect_intermediates=collect_intermediates)
        output = attn_output.pop("output")
        output = torch.cat(
            [
                output,
                torch.zeros(
                    output.shape[0],
                    output.shape[1],
                    num_total - output.shape[2],
                    device=output.device,
                ),
            ],
            dim=2,
        )
        return attn_output, output


class MultiRoundMotifInfluenceTanhLongRangeProcessor(nn.Module):
    """
    Uses the motifs to influence the outputs.
    """

    def __init__(
        self,
        num_motifs,
        cl,
        *,
        propagate_sparsity_spec,
        num_heads=1,
        motif_influence_rounds=1,
        max_len=10_000,
        forward_only=False,
        num_outputs=2,
        embed_dim=128,
        tanh_attn_layer_spec=dict(type="TanhConvolvedAttnLayer"),
    ):
        super().__init__()

        self.num_outputs = num_outputs

        self.attns = nn.ModuleList(
            [
                TanhConvolvedAttnOnIndices(
                    total_dim=num_motifs - num_outputs,
                    input_idxs=list(range(num_motifs - num_outputs)),
                    output_idxs=list(range(num_motifs - num_outputs)),
                    embed_dim=embed_dim,
                    num_outputs=num_motifs - num_outputs,
                    window=cl,
                    num_heads=num_heads,
                    forward_only=forward_only,
                    tanh_attn_layer_spec=tanh_attn_layer_spec,
                    max_len=max_len,
                )
                for _ in range(motif_influence_rounds)
            ]
        )

        self.propagate_sparsity = construct(
            sparsity_propagation_types(),
            propagate_sparsity_spec,
        )

        self.single_round = SingleTanhLongRangeProcessor(
            num_motifs,
            cl,
            num_heads=num_heads,
            max_len=max_len,
            forward_only=forward_only,
            num_outputs=num_outputs,
            embed_dim=embed_dim,
        )

    def single_round_of_motif_influence(self, attn, motifs, collect_intermediates):
        infl_output = attn(motifs, collect_intermediates=collect_intermediates)
        infl = infl_output.pop("output")
        infl = torch.sigmoid(infl)
        motifs = self.propagate_sparsity(infl, motifs)
        return infl_output, motifs

    def motif_influences(self, output, collect_intermediates):
        attn_outputs = []
        motifs = output
        for attn in self.attns:
            attn_output, motifs = self.single_round_of_motif_influence(
                attn, motifs, collect_intermediates
            )
            attn_outputs.append(attn_output)
        return attn_outputs, motifs

    def forward(self, output, collect_intermediates):
        motifs = output[:, :, self.num_outputs :]
        motif_attn_outputs, motifs = self.motif_influences(
            motifs, collect_intermediates
        )
        output = torch.cat([output[:, :, : self.num_outputs], motifs], dim=2)
        splicing_attn_output, output = self.single_round(output, collect_intermediates)
        attn_output = dict(
            motif_attn_outputs=motif_attn_outputs,
            splicing_attn_output=splicing_attn_output,
        )
        return attn_output, output
