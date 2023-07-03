import torch
import torch.nn as nn

from modular_splicing.models.entire_model.modular_predictor import (
    ModularSplicePredictor,
)


def compute_setting_as_batch(input_dict):
    setting = input_dict["setting"]
    setting_batch = setting[:, 0, 0]
    assert (setting == setting_batch[:, None, None]).all().item()
    return setting_batch


class MultiTissueProbabilitiesPredictorEnv(ModularSplicePredictor):
    def __init__(self, *, num_tissues, num_motifs, **kwargs):
        super().__init__(num_motifs=num_motifs, **kwargs)
        self.num_tissues = num_tissues
        self.environment_per_tissue = nn.Parameter(
            torch.randn(num_tissues, num_motifs - 2)
        )

    def apply_environment(self, x, input_dict):
        setting_batch = compute_setting_as_batch(input_dict)
        adjustments = torch.sigmoid(self.environment_per_tissue[setting_batch])
        return x * adjustments[:, None]


class MultiTissueProbabilitiesPredictorEnvPresparse(
    MultiTissueProbabilitiesPredictorEnv
):
    def run_pre_sparse(self, x, input_dict):
        return self.apply_environment(x, input_dict)


class MultiTissueProbabilitiesPredictorEnvPostsparse(
    MultiTissueProbabilitiesPredictorEnv
):
    def run_post_sparse(self, x, input_dict):
        spl = x[:, :, :2]
        x = x[:, :, 2:]
        x = self.apply_environment(x, input_dict)
        x = torch.cat([spl, x], dim=2)
        return x


class MultiTissueProbabilitiesPredictorMultiAggregator(ModularSplicePredictor):
    """
    Version of ModularSplicePredictor that has multiple aggregators, one for each
        tissue. The aggregators are all identical in architecture, but have different
        parameters.
    """

    motif_model_modules = ["splicepoint_model", "motif_model", "sparsity_enforcer"]
    aggregator_modules = [
        "influence_calculator",
        "propagate_sparsity",
        "final_processor",
        "final_polish",
    ]

    def __init__(self, *, num_tissues, **kwargs):
        super().__init__(**kwargs)
        assert set(self.motif_model_modules) | set(self.aggregator_modules) == set(
            self._modules.keys()
        )
        assert set(self.motif_model_modules) & set(self.aggregator_modules) == set()
        self.num_tissues = num_tissues
        self.aggregators = nn.ModuleList(
            [ModularSplicePredictor(**kwargs) for _ in range(num_tissues)]
        )
        for aggregator_module in self.aggregator_modules:
            del self._modules[aggregator_module]
        for agg in self.aggregators:
            for motif_model_module in self.motif_model_modules:
                del agg._modules[motif_model_module]

    def forward_post_motifs(
        self, post_sparse, splicepoint_results_residual, input_dict=None, **kwargs
    ):
        setting_batch = compute_setting_as_batch(input_dict)
        settings = sorted(set(setting_batch.tolist()))
        outs = []
        for setting in settings:
            mask = setting_batch == setting
            ps = post_sparse[mask]
            sp = splicepoint_results_residual[mask]
            out = self.aggregators[setting].forward_post_motifs(
                ps, sp, input_dict=input_dict, **kwargs
            )
            outs.append(out)
        if isinstance(outs[0], dict):
            assert all(isinstance(out, dict) for out in outs)
            assert all(set(out.keys()) == set(outs[0].keys()) for out in outs)
            return {
                k: self.combine_arrays(
                    settings, setting_batch, [out[k] for out in outs]
                )
                for k in outs[0].keys()
            }
        return self.combine_arrays(settings, setting_batch, outs)

    def combine_arrays(self, settings, setting_batch, outs):
        assert all(out.shape[1:] == outs[0].shape[1:] for out in outs)
        result = torch.zeros(
            setting_batch.shape[0], *outs[0].shape[1:], device=outs[0].device
        )
        for setting, out in zip(settings, outs):
            mask = setting_batch == setting
            result[mask] = out
        return result


def multi_tissue_probabilities_predictor_env_types():
    return dict(
        MultiTissueProbabilitiesPredictorEnvPresparse=MultiTissueProbabilitiesPredictorEnvPresparse,
        MultiTissueProbabilitiesPredictorEnvPostsparse=MultiTissueProbabilitiesPredictorEnvPostsparse,
        MultiTissueProbabilitiesPredictorMultiAggregator=MultiTissueProbabilitiesPredictorMultiAggregator,
    )
