"""
Torch loading with the ability to remap symbols (corresponding to torch.nn.Module objects).

This is necessary since we have moved several classes around in the codebase,
    and we need to be able to load old models. By default, torch.load will
    fail if it encounters a class that is not in the same place as it was when
    the model was saved.

The only function you need to use is load_with_remapping_pickle, which is a
    drop-in replacement for torch.load.
"""

import pickle

from permacache import permacache, swap_unpickler_context_manager

# Remaps modules to their new locations.
# Only use if the entire module has been moved.
MODULE_RENAME_MAP = {
    "auto_minimize_motifs.motif_selector": "shelved.auto_minimize_motifs.motif_selector",
    "branch_sites.splicepoint_based_branch_site_predictor": "shelved.branch_sites.splicepoint_based_branch_site_predictor",
    "binary_motif_model": "modular_splicing.fit_rbns.chenxi_neural_model",
    "eclip_analysis.eclip_model": "modular_splicing.eclip.trained_on_eclip.model",
    "eclip_analysis.train_models": "modular_splicing.eclip.trained_on_eclip.train",
    "rbns_data": "modular_splicing.fit_rbns.rbns_data",
    "donorlike_motifs.donor_separating_motif": "shelved.donorlike_motifs.donor_separating_motif",
    "splice_point_identifier": "modular_splicing.models.lssi",
    "modules.multi_motif_parallel_sparsity_enforcer": "shelved.robustly_adjusted.multi_motif_parallel_sparsity_enforcer",
    "modules.local_splicepoint_model_residual_propagator": "modular_splicing.models.local_splicepoint_model_residual_propagator",
    "modules.robustly_adjusted_motifs": "shelved.robustly_adjusted.robustly_adjusted_motifs",
    "modules.motif_model_multiple_width": "shelved.motif_model_multiple_width",
    "modules.compute_extra_input_then": "modular_splicing.models.motif_models.compute_extra_input_then",
    "modules.use_inputs_in": "modular_splicing.models.motif_models.use_inputs_in",
    "modules.spliceai_as_downstream": "modular_splicing.models.entire_model.spliceai_as_downstream",
    "modules.reconstruct_sequence": "modular_splicing.models.entire_model.reconstruct_sequence",
    "multiheadattention": "modular_splicing.models.modules.conv_attn.multiheadattention",
    "convolved_attn": "modular_splicing.models.modules.conv_attn.convolved_attn",
    "short_range_motif_reprocessor": "modular_splicing.models.motif_models.reprocessed_motif_model.short_range_motif_reprocessor",
}
# Remaps (module, symbol) pairs to their new (module, symbol) locations.
# Neither the old nor new module should be in MODULE_RENAME_MAP.
SYMBOL_RENAME_MAP = {
    ("torch.nn.modules.linear", "_LinearWithBias"): (
        "torch.nn.modules.linear",
        "NonDynamicallyQuantizableLinear",
    ),
    ("motifs", "TorchPSAM"): ("modular_splicing.psams.psams", "TorchPSAM"),
    ("spliceai_torch", "ResidualUnit"): (
        "modular_splicing.models.modules.residual_unit",
        "ResidualUnit",
    ),
    ("spliceai_torch", "HalfResidualUnit"): (
        "modular_splicing.models.modules.residual_unit",
        "HalfResidualUnit",
    ),
    ("spliceai_torch", "SpatiallySparse"): (
        "modular_splicing.models.modules.sparsity.spatially_sparse_by_channel",
        "SpatiallySparseByChannel",
    ),
    ("spliceai_torch", "NoSparsity"): (
        "modular_splicing.models.modules.sparsity.no_sparsity",
        "NoSparsity",
    ),
    ("spliceai_torch", "ParallelSpatiallySparse"): (
        "modular_splicing.models.modules.sparsity.parallel_spatially_sparse",
        "ParallelSpatiallySparse",
    ),
    ("spliceai_torch", "SpatiallySparseAcrossChannels"): (
        "modular_splicing.models.modules.sparsity.spatially_sparse_across_channels",
        "SpatiallySparseAcrossChannels",
    ),
    ("spliceai_torch", "SpatiallySparseAcrossChannelsDropMotifs"): (
        "modular_splicing.models.modules.sparsity.spatially_sparse_across_channels_drop_motifs",
        "SpatiallySparseAcrossChannelsDropMotifs",
    ),
    ("spliceai_torch", "DropMotifsIn"): (
        "modular_splicing.models.modules.sparsity.drop_motifs_in",
        "DropMotifsIn",
    ),
    ("spliceai_torch", "DiscretizeMotifsIn"): (
        "modular_splicing.models.modules.sparsity.discretize_motifs_in",
        "DiscretizeMotifsIn",
    ),
    ("spliceai_torch", "SpliceAI"): (
        "modular_splicing.models.modules.spliceai",
        "SpliceAI",
    ),
    ("spliceai_torch", "no_preprocess"): (
        "modular_splicing.models.modules.spliceai",
        "no_preprocess",
    ),
    ("spliceai_torch", "ensemble"): (
        "modular_splicing.models.modules.ensemble",
        "ensemble",
    ),
    ("modular_splice_predictor", "ResidualStack"): (
        "modular_splicing.models.modules.residual_unit_stack",
        "ResidualStack",
    ),
    ("modular_splice_predictor", "FCMotifFeatureExtractor"): (
        "modular_splicing.models.modules.fc_motif_feature_extractor",
        "FCMotifFeatureExtractor",
    ),
    ("modular_splice_predictor", "SplicepointModel"): (
        "modular_splicing.models.modules.lssi_in_model",
        "BothLSSIModels",
    ),
    ("modular_splice_predictor", "SplicepointModelDummy"): (
        "modular_splicing.models.modules.lssi_in_model",
        "BothLSSIModelsDummy",
    ),
    ("modular_splice_predictor", "MotifModel"): (
        "modular_splicing.models.motif_models.learned_motif_model",
        "LearnedMotifModel",
    ),
    ("modular_splice_predictor", "NoMotifModel"): (
        "modular_splicing.models.motif_models.no_motif_model",
        "NoMotifModel",
    ),
    ("modular_splice_predictor", "ParallelMotifModels"): (
        "modular_splicing.models.motif_models.parallel_motif_model",
        "ParallelMotifModels",
    ),
    ("modular_splice_predictor", "AdjustedMotifModel"): (
        "modular_splicing.models.motif_models.adjusted_motif_model",
        "AdjustedMotifModel",
    ),
    ("modular_splice_predictor", "ReprocessedMotifModel"): (
        "modular_splicing.models.motif_models.reprocessed_motif_model.reprocessed_motifs",
        "ReprocessedMotifModel",
    ),
    ("modules.flanked_motif_model", "FlankedMotifModel"): (
        "modular_splicing.models.motif_models.flanked_motif_model",
        "FlankedMotifModel",
    ),
    ("modular_splice_predictor", "AdjustedSplicepointModel"): (
        "shelved.donorlike_motifs.models",
        "AdjustedSplicepointModel",
    ),
    ("modular_splice_predictor", "FromLoadedSplicepointModel"): (
        "shelved.donorlike_motifs.models",
        "FromLoadedSplicepointModel",
    ),
    ("modular_splice_predictor", "PretrainedRBNSAdjustedMotifModel"): (
        "modular_splicing.models.motif_models.neural_fixed_motif_model",
        "NeuralFixedMotif",
    ),
    ("modular_splice_predictor", "PSAMMotifModel"): (
        "modular_splicing.models.motif_models.psam_fixed_motif",
        "PSAMMotifModel",
    ),
    ("modular_splice_predictor", "MotifModelForRobustDownstream"): (
        "shelved.uniqueness.models",
        "MotifModelForRobustDownstream",
    ),
    ("modular_splice_predictor", "SingleAttentionLongRangeProcessor"): (
        "modular_splicing.models.modules.influence_value.single_attention_long_range",
        "SingleAttentionLongRangeProcessor",
    ),
    ("modular_splice_predictor", "InfluenceValueCalculator"): (
        "modular_splicing.models.modules.influence_value.influence_value_calculator",
        "InfluenceValueCalculator",
    ),
    ("modular_splice_predictor", "SparsityEnforcer"): (
        "modular_splicing.models.modules.sparsity_enforcer.sparsity_enforcer",
        "SparsityEnforcer",
    ),
    ("modular_splice_predictor", "NoPresparseNormalizer"): (
        "modular_splicing.models.modules.sparsity_enforcer.presparse_normalizer",
        "NoPresparseNormalizer",
    ),
    ("modular_splice_predictor", "BasicPresparseNormalizer"): (
        "modular_splicing.models.modules.sparsity_enforcer.presparse_normalizer",
        "BasicPresparseNormalizer",
    ),
    ("modular_splice_predictor", "LSTMLongRangeFinalLayer"): (
        "modular_splicing.models.modules.final_layer.lstm_final_layer",
        "LSTMLongRangeFinalLayer",
    ),
    ("modular_splice_predictor", "FinalProcessor"): (
        "modular_splicing.models.modules.final_layer.final_processor",
        "FinalProcessor",
    ),
    ("modular_splice_predictor", "ProductSparsityPropagation"): (
        "modular_splicing.models.modules.sparsity_propagation",
        "ProductSparsityPropagation",
    ),
    ("modular_splice_predictor", "NoSparsityPropagation"): (
        "modular_splicing.models.modules.sparsity_propagation",
        "NoSparsityPropagation",
    ),
    ("modular_splice_predictor", "ModularSplicePredictor"): (
        "modular_splicing.models.entire_model.modular_predictor",
        "ModularSplicePredictor",
    ),
    ("modular_splice_predictor", "SeparateDownstreams"): (
        "modular_splicing.models.entire_model.separate_downstreams",
        "SeparateDownstreams",
    ),
    ("modular_splice_predictor", "SpliceAIModule"): (
        "modular_splicing.models.modules.spliceai",
        "SpliceAIModule",
    ),
    ("modular_splice_predictor", "PreprocessedSpliceAI"): (
        "modular_splicing.models.entire_model.preprocessed_spliceai",
        "PreprocessedSpliceAI",
    ),
    ("modular_splice_predictor", "PassthroughMotifsFromData"): (
        "modular_splicing.models.motif_models.passthrough_motif_model",
        "PassthroughMotifsFromData",
    ),
    ("utils", "PositionalEncoding"): (
        "modular_splicing.models.modules.positional_encoding",
        "PositionalEncoding",
    ),
    ("residual_motifs", "NoResidualMotifs"): (
        "modular_splicing.models.motif_models.learned_motif_model",
        "NoResidualMotifs",
    ),
    ("residual_motifs", "FixedMotifs"): (
        "modular_splicing.models.motif_models.psam_fixed_motif",
        "FixedMotifs",
    ),
    ("residual_motifs", "RbnsPsamMotifs"): (
        "modular_splicing.models.motif_models.psam_fixed_motif",
        "RbnsPsamMotifs",
    ),
    ("utils", "AdaptiveSparsityThresholdManager"): (
        "modular_splicing.train.adaptive_sparsity_threshold_manager",
        "AdaptiveSparsityThresholdManager",
    ),
    ("utils", "Sparse"): ("modular_splicing.utils.arrays", "Sparse"),
}


class renamed_symbol_unpickler(pickle.Unpickler):
    """
    Unpicler that renames modules and symbols as specified in the
    MODULE_RENAME_MAP and SYMBOL_RENAME_MAP dictionaries.
    """

    def find_class(self, module, name):
        if (module, name) in SYMBOL_RENAME_MAP:
            assert module not in MODULE_RENAME_MAP
            module, name = SYMBOL_RENAME_MAP[(module, name)]
            assert module not in MODULE_RENAME_MAP
        if module in MODULE_RENAME_MAP:
            module = MODULE_RENAME_MAP[module]

        try:
            return super(renamed_symbol_unpickler, self).find_class(module, name)
        except:
            print("Could not find", (module, name))
            raise


class remapping_pickle:
    """
    An instance of this class will behave like the pickle module, but
    will use the renamed_symbol_unpickler class instead of the default
    Unpickler class.
    """

    def __getattribute__(self, name):
        if name == "Unpickler":
            return renamed_symbol_unpickler
        return getattr(pickle, name)

    def __hasattr__(self, name):
        return hasattr(pickle, name)


def load_with_remapping_pickle(*args, **kwargs):
    """
    Behaves like torch.load, but re-maps modules.
    """
    import torch

    return torch.load(*args, **kwargs, pickle_module=remapping_pickle())


def permacache_with_remapping_pickle(*args, **kwargs):
    """
    Behaves like permacache.permacache, but re-maps modules on load.
    """
    return permacache(
        *args,
        **kwargs,
        read_from_shelf_context_manager=swap_unpickler_context_manager(
            remapping_pickle().Unpickler
        ),
    )
