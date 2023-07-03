import torch.nn as nn

from modular_splicing.utils.construct import construct

from modular_splicing.models.modules.lssi_in_model import both_lssi_model_types
from modular_splicing.models.motif_models.types import motif_model_types
from modular_splicing.models.local_splicepoint_model_residual_propagator import (
    EarlyChannelLSMRP,
    lsmrp_types,
)
from modular_splicing.models.modules.final_layer.final_processor import FinalProcessor
from modular_splicing.models.modules.influence_value.influence_value_calculator import (
    influence_value_calculator_types,
)
from modular_splicing.models.modules.sparsity_propagation import (
    sparsity_propagation_types,
)
from modular_splicing.models.modules.sparsity_enforcer.sparsity_enforcer import (
    sparsity_enforcer_types,
)


class ModularSplicePredictor(nn.Module):
    """
    Full model.

    Parameters
    ----------
    num_motifs : int
        Number of motifs to use.
    input_size : int
        Size of input.
    output_size : int
        Size of output.
    cl : int
        Relationship between input and output sizes, output has cl/2 cut off each side.
    sparsity : float
        Sparsity of the model.
    sparse_spec : dict
        Specification of the sparse layer.
    motif_model_spec : dict
        Specification of the motif model.
    sparsity_enforcer_spec : dict
        Specification of the sparsity enforcer.
    influence_calculator_spec : dict
        Specification of the influence calculator.
    propagate_sparsity_spec : dict
        Specification of the sparsity propagation.
    final_polish_spec : dict
        Specification of the final polish.
    lsmrp_spec : dict
        Specification of the LSMRP.
    final_processor_spec : dict
        Specification of the final processor.
    channels : int
        Number of channels to use in various internal layers.
    splicepoint_model_spec : dict
        Specification of the splicepoint model.
    affine_sparsity_enforcer : bool
        Whether to use an affine sparsity enforcer.
    """

    _input_dictionary = True

    def __init__(
        self,
        *,
        num_motifs,
        input_size=4,
        output_size=3,
        cl=80,
        sparsity,
        sparse_spec,
        motif_model_spec,
        sparsity_enforcer_spec=dict(type="SparsityEnforcer"),
        influence_calculator_spec,
        propagate_sparsity_spec=dict(type="ProductSparsityPropagation"),
        final_polish_spec=dict(type="Identity"),
        lsmrp_spec=dict(type="EarlyChannelLSMRP"),
        final_processor_spec,
        channels,
        splicepoint_model_spec,
        affine_sparsity_enforcer=False,
    ):
        super().__init__()

        self.splicepoint_model = construct(
            both_lssi_model_types(), splicepoint_model_spec
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

        self.influence_calculator = construct(
            influence_value_calculator_types(),
            influence_calculator_spec,
            num_motifs=num_motifs,
            channels=channels,
            cl=cl,
        )

        self.propagate_sparsity = construct(
            sparsity_propagation_types(),
            propagate_sparsity_spec,
        )

        self.final_processor = construct(
            dict(FinalProcessor=FinalProcessor),
            final_processor_spec,
            num_motifs=num_motifs,
            channels=channels,
            output_size=output_size,
        )

        self.propagate_residuals = construct(
            lsmrp_types(),
            lsmrp_spec,
        )

        self.final_polish = construct(dict(Identity=nn.Identity), final_polish_spec)

        self.cl = cl

    @property
    def sparse_layer(self):
        return self.sparsity_enforcer.sparse_layer

    def update_sparsity(self, update_by):
        self.sparse_layer.update_sparsity(update_by)
        self.motif_model.notify_sparsity(self.get_sparsity())

    def get_sparsity(self):
        return self.sparse_layer.get_sparsity()

    def run_pre_sparse(self, motif_model_output, input_dict):
        """
        Override this to run something before the sparse layer.
        """
        return motif_model_output

    def run_post_sparse(self, sparse_output, input_dict):
        """
        Override this to run something after the sparse layer.
        """
        return sparse_output

    def forward(
        self,
        input_dict,
        collect_intermediates=False,
        collect_losses=False,
        only_motifs=False,
        manipulate_post_sparse=None,
        manipulate_splicepoint_motif=None,
    ):
        """
        Run the model.

        Parameters
        ----------
        input_dict : dict
            Dictionary of inputs.
        collect_intermediates : bool
            Whether to collect intermediate values.
        collect_losses : bool
            Whether to collect losses.
        only_motifs : bool
            Whether to only run the motif model.
        manipulate_post_sparse : callable
            Function to manipulate the output of the sparse layer.
        """
        # for backwards compat, in case the input is a tensor
        if not isinstance(input_dict, dict):
            input_dict = dict(x=input_dict)
        input = input_dict["x"]

        # have to collect intermediates to get the motifs
        if only_motifs:
            collect_intermediates = True

        full_outputs = {}

        def add(**kwargs):
            if collect_intermediates:
                full_outputs.update(kwargs)

        splicepoint_results, splicepoint_results_residual = self.splicepoint_model(
            input, manipulate_splicepoint_motif=manipulate_splicepoint_motif
        )

        add(splicepoint_results=splicepoint_results)
        add(splicepoint_results_residual=splicepoint_results_residual)

        motif_model_output = self.motif_model(
            input_dict
            if getattr(self.motif_model, "_input_is_dictionary", False)
            else input
        )

        if not getattr(self.motif_model, "motif_model_dict", False):
            motif_model_output = dict(motifs=motif_model_output)
        output = motif_model_output.pop("motifs")
        full_outputs.update(motif_model_output)

        output = self.run_pre_sparse(output, input_dict)

        full_outputs_enforcer, output = self.sparsity_enforcer(
            output,
            splicepoint_results,
            manipulate_post_sparse,
            collect_intermediates,
            **motif_model_output.get("sparsity_enforcer_extra_params", {}),
        )

        output = self.run_post_sparse(output, input_dict)

        full_outputs.update(full_outputs_enforcer)

        if only_motifs:
            return full_outputs

        post_motifs_result = self.forward_post_motifs(
            output,
            splicepoint_results_residual,
            collect_intermediates=collect_intermediates,
            collect_losses=collect_losses,
            input_dict=input_dict,
        )

        full_outputs.update(post_motifs_result)
        if collect_intermediates or collect_losses:
            return full_outputs
        else:
            return full_outputs["output"]

    def forward_post_motifs(
        self,
        post_sparse,
        splicepoint_results_residual,
        *,
        collect_intermediates,
        collect_losses,
        input_dict=None,
    ):
        # input_dict is not necessary here but is passed in case it is useful
        full_outputs = {}

        def add(**kwargs):
            if collect_intermediates:
                full_outputs.update(kwargs)

        to_influence = post_sparse

        attn_output, influence = self.influence_calculator(
            post_sparse, collect_intermediates
        )

        add(influence=influence)
        output = self.propagate_sparsity(influence, to_influence)
        add(post_influence=output)

        final_processor_output = self.final_processor(
            output, splicepoint_results_residual
        )

        output = final_processor_output.pop("output")

        if collect_losses:
            full_outputs.update(final_processor_output)

        output = getattr(
            self, "propagate_residuals", EarlyChannelLSMRP()
        ).propagate_residuals(output, splicepoint_results_residual)

        output = self.final_polish(output)

        output = output[:, self.cl // 2 : output.shape[1] - self.cl // 2, :]

        result = dict(output=output, **full_outputs, **attn_output)

        for k in list(result.keys()):
            if k.startswith("output_to_evaluate"):
                result[k] = result[k][
                    :, self.cl // 2 : -self.cl // 2, : self.final_processor.output_size
                ]

        if collect_intermediates:
            return result
        else:
            return {
                k: v
                for k, v in result.items()
                if k == "output"
                or k.startswith("output_to_evaluate")
                or k.startswith("weight_of_output_to_evaluate")
            }
