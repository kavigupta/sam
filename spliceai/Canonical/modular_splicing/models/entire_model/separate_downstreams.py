import torch
import torch.nn as nn

from .entire_model_types import entire_model_types

from modular_splicing.utils.construct import construct


class SeparateDownstreams(nn.Module):
    """
    Have several different downstreams and only one motif model.

    Each downstream corresponds to a different output.

    These are each specified in the `pattern` list.

    The pattern list is a list of integers, where each integer
    corresponds to a downstream.  The integer is the index of the
    channel in the output of the downstream that should be used
    for the corresponding channel in the output of this model.

    E.g., if the pattern is [None, 1, 2], then the first channel
    of the output of this model will be all zeros, the second
    channel will be the second channel of the first downstream,
    and the third channel will be the third channel of the second
    downstream.
    """

    def __init__(
        self,
        pattern=[None, 1, 2],
        *,
        entire_model_specs,
        **kwargs,
    ):
        super().__init__()
        assert all(
            entire_model_spec["type"] == "ModularSplicePredictor"
            for entire_model_spec in entire_model_specs
        )
        assert len(entire_model_specs) == sum(x is not None for x in pattern)
        self.models = nn.ModuleList(
            [
                construct(
                    entire_model_types(),
                    entire_model_spec,
                    output_size=len(pattern),
                    **kwargs,
                )
                for entire_model_spec in entire_model_specs
            ]
        )
        self.pattern = pattern
        for model in self.models[1:]:
            del model.splicepoint_model
            del model.motif_model
            del model.sparsity_enforcer

    def forward(
        self,
        input_dict,
        only_motifs=False,
        collect_intermediates=False,
        collect_losses=False,
    ):
        first_part = self.models[0](
            input_dict,
            only_motifs=True,
            collect_intermediates=collect_intermediates,
            collect_losses=collect_losses,
        )
        if only_motifs:
            return first_part
        out_each = [
            mod.forward_post_motifs(
                post_sparse=first_part["post_sparse"],
                splicepoint_results_residual=first_part["splicepoint_results_residual"],
                collect_intermediates=collect_intermediates,
            )
            for mod in self.models
        ]

        out_each = [out["output"].log_softmax(-1) for out in out_each]

        zero_value = torch.zeros_like(out_each[0][:, :, 0])

        results = []
        i = 0
        for el in self.pattern:
            if el is None:
                results.append(zero_value)
            else:
                results.append(out_each[i][:, :, el])
                i += 1
        assert i == len(out_each)

        results = torch.stack(results).permute(1, 2, 0)
        results = results.log_softmax(-1)
        if collect_intermediates or collect_losses:
            return dict(output=results)
        else:
            return results

    @property
    def sparse_layer(self):
        return self.models[0].sparse_layer

    def update_sparsity(self, update_by):
        return self.models[0].update_sparsity(update_by)

    def get_sparsity(self):
        return self.models[0].get_sparsity()
