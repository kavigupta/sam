import torch.nn as nn
from modular_splicing.models.modules.spliceai import SpliceAIModule
from modular_splicing.models.modules.sparsity_enforcer.sparsity_enforcer import (
    sparsity_enforcer_types,
)
from modular_splicing.models.motif_models.types import motif_model_types
from modular_splicing.models.modules.lssi_in_model import both_lssi_model_types
from modular_splicing.models.local_splicepoint_model_residual_propagator import (
    lsmrp_types,
)

from modular_splicing.utils.construct import construct


class SpliceaiAsDownstream(nn.Module):
    """
    Similar to `ModularSplicePredictor`, but uses a `SpliceAI` module as the downstream module.

    Args:
        motif_model_spec (dict): Specification for the motif model.
        spliceai_spec (dict): Specification for the `SpliceAI` module.
        splicepoint_model_spec (dict): Specification for the LSSI model.
        sparsity_enforcer_spec (dict): Specification for the sparsity enforcer.
        sparse_spec (dict): Specification for the sparsity layer.
        input_size (int): Size of the input.
        sparsity (float): Initial sparsity of the sparsity layer.
        cl (int): Length of the Spliceai's context window.
        num_motifs (int): Number of motifs.
        channels (int): Number of channels in the motif model.
        lsmrp_spec (dict): Specification for the local splicepoint model residual propagator.
        affine_sparsity_enforcer (bool): Whether to use affine sparsity enforcer.
    """

    def __init__(
        self,
        *,
        motif_model_spec,
        spliceai_spec,
        splicepoint_model_spec,
        sparsity_enforcer_spec=dict(type="SparsityEnforcer"),
        sparse_spec,
        input_size=4,
        sparsity=None,
        cl,
        num_motifs,
        channels,
        lsmrp_spec=dict(type="EarlyChannelLSMRP"),
        affine_sparsity_enforcer=False,
    ):
        super().__init__()
        assert input_size == 4
        self.splicepoint_model = construct(
            both_lssi_model_types(),
            splicepoint_model_spec,
        )

        self.motif_model = construct(
            motif_model_types(),
            motif_model_spec,
            input_size=input_size,
            channels=channels,
            num_motifs=num_motifs - 2,
        )

        self.sparsity_enforcer = construct(
            sparsity_enforcer_types(),
            sparsity_enforcer_spec,
            num_motifs=num_motifs - 2,
            sparse_spec=sparse_spec,
            sparsity=sparsity,
            affine=affine_sparsity_enforcer,
        )
        self.spliceai = construct(
            dict(SpliceAIModule=SpliceAIModule),
            spliceai_spec,
            window=cl,
            input_size=num_motifs,
        )
        self.propagate_residuals = construct(
            lsmrp_types(),
            lsmrp_spec,
        )

    @property
    def cl(self):
        return self.spliceai.spliceai.cl

    @property
    def sparse_layer(self):
        return self.sparsity_enforcer.sparse_layer

    def forward(
        self,
        x,
        collect_intermediates=False,
        collect_losses=False,
        only_motifs=False,
        manipulate_post_sparse=None,
    ):
        if isinstance(x, dict):
            x = x["x"]
        splicepoint_results, splicepoint_results_residual = self.splicepoint_model(x)

        splicepoint_results_residual = splicepoint_results_residual[
            :, self.cl // 2 : splicepoint_results_residual.shape[1] - self.cl // 2
        ]

        full_outputs = {}

        full_outputs["splicepoint_results_residual"] = splicepoint_results_residual

        motif_model_output = self.motif_model(x)
        if not getattr(self.motif_model, "motif_model_dict", False):
            motif_model_output = dict(motifs=motif_model_output)
        x = motif_model_output.pop("motifs")
        full_outputs.update(motif_model_output)

        full_outputs_enforcer, x = self.sparsity_enforcer(
            x,
            splicepoint_results,
            manipulate_post_sparse,
            collect_intermediates or only_motifs,
            **motif_model_output.get("sparsity_enforcer_extra_params", {}),
        )

        full_outputs.update(full_outputs_enforcer)

        if only_motifs:
            return full_outputs
        x = self.spliceai(x)
        x = self.propagate_residuals.propagate_residuals(
            x, splicepoint_results_residual
        )
        if collect_intermediates or collect_losses:
            return dict(output=x, **full_outputs)
        else:
            return x

    def update_sparsity(self, update_by):
        self.sparse_layer.update_sparsity(update_by)
        self.motif_model.notify_sparsity(self.get_sparsity())

    def get_sparsity(self):
        return self.sparse_layer.get_sparsity()
