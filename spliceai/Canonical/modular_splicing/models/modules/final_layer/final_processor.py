import torch
import torch.nn as nn

from modular_splicing.utils.construct import construct

from modular_splicing.models.modules.residual_unit_stack import ResidualStack
from modular_splicing.models.local_splicepoint_model_residual_propagator import (
    lsmrp_types,
)
from .bottleneck import bottleneck_types
from .lstm_final_layer import LSTMLongRangeFinalLayer


class FinalProcessor(nn.Module):
    """
    Runs a post-influence layer, followed by a long-range final layer, and then a projection layer.

    You can increase the number of long range final layer channels by setting long_range_in_channels
    and long_range_out_channels. If you don't set these, they default to num_motifs.
    """

    def __init__(
        self,
        *,
        bottleneck_spec=dict(type="Identity"),
        post_bottleneck_lsmrp_spec=None,
        evaluate_post_bottleneck=False,
        evaluate_post_bottleneck_weight=1,
        post_influence_spec,
        long_range_final_layer_spec=dict(type="LSTMLongRangeFinalLayer"),
        num_motifs,
        output_size,
        channels,
        long_range_in_channels=None,
        long_range_out_channels=None,
    ):
        super().__init__()

        assert evaluate_post_bottleneck in [True, False]

        if long_range_in_channels is None:
            long_range_in_channels = num_motifs

        if long_range_out_channels is None:
            long_range_out_channels = num_motifs

        self.bottleneck = construct(
            bottleneck_types(),
            bottleneck_spec,
            num_motifs=num_motifs,
        )

        self.evaluate_post_bottleneck = evaluate_post_bottleneck
        self.evaluate_post_bottleneck_weight = evaluate_post_bottleneck_weight
        if post_bottleneck_lsmrp_spec is not None:
            self.post_bottleneck_lsmrp = construct(
                lsmrp_types(), post_bottleneck_lsmrp_spec
            )
        else:
            assert not evaluate_post_bottleneck

        self.post_influence_reprocessor = construct(
            dict(ResidualStack=ResidualStack),
            post_influence_spec,
            input_channels=num_motifs,
            hidden_channels=channels,
            output_channels=long_range_in_channels,
        )

        self.long_range_final = construct(
            dict(LSTMLongRangeFinalLayer=LSTMLongRangeFinalLayer),
            long_range_final_layer_spec,
            input_channels=long_range_in_channels,
            output_channels=long_range_out_channels,
        )

        self.output_projection = nn.Conv1d(long_range_out_channels, output_size, 1)

    def forward(self, output, splicepoint_results_residual):
        if hasattr(self, "bottleneck"):
            output = self.bottleneck(output)

        full_outputs = {}

        post_bottleneck_lsmrp = getattr(self, "post_bottleneck_lsmrp", None)
        if post_bottleneck_lsmrp is not None:
            padding = torch.zeros(
                (output.shape[0], output.shape[1], 1),
                device=output.device,
                dtype=output.dtype,
            )

            output = torch.cat([padding, output], axis=2)
            output = post_bottleneck_lsmrp.propagate_residuals(
                output, splicepoint_results_residual
            )

            if self.evaluate_post_bottleneck:
                full_outputs["output_to_evaluate (alternate post bottleneck)"] = (
                    output * self.evaluate_post_bottleneck
                )
                full_outputs[
                    "weight_of_output_to_evaluate (alternate post bottleneck)"
                ] = getattr(self, "evaluate_post_bottleneck_weight", 1)
            output = output[:, :, 1:]

        output = self.post_influence_reprocessor(output)
        output = self.long_range_final(output)

        output = output.transpose(1, 2)
        output = self.output_projection(output)
        output = output.transpose(1, 2)
        full_outputs["output"] = output
        return full_outputs

    @property
    def output_size(self):
        return self.output_projection.out_channels
