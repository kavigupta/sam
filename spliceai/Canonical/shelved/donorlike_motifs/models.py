import torch
import torch.nn as nn
from modular_splicing.models.entire_model.modular_predictor import (
    ModularSplicePredictor,
)

from modular_splicing.models.motif_models.types import motif_model_types
from modular_splicing.models.modules.lssi_in_model import (
    LSSI_MODEL_THRESH,
    load_individual_lssi_model,
)
from modular_splicing.utils.construct import construct


class AdjustedSplicepointModel(nn.Module):
    motif_model_dict = True

    def __init__(
        self,
        splicepoint_model_path,
        splicepoint_index,
        adjustment_model_spec,
        **kwargs,
    ):
        super().__init__()
        self.model_to_adjust = load_individual_lssi_model(
            splicepoint_model_path, trainable=False
        )
        self.adjustment_model = construct(
            motif_model_types(), adjustment_model_spec, **kwargs
        )
        self.splicepoint_index = splicepoint_index

    def forward(self, x):
        to_adjust = self.model_to_adjust(x).log_softmax(-1)
        to_adjust = to_adjust[:, :, [self.splicepoint_index]]
        to_adjust -= LSSI_MODEL_THRESH
        to_adjust = torch.nn.functional.relu(to_adjust)
        adjustment = self.adjustment_model(x)
        adjustment = adjustment * (to_adjust != 0).float()
        adjusted = to_adjust + adjustment
        return dict(motifs=adjusted, pre_adjustment_motifs=to_adjust)

    def notify_sparsity(self, sparsity):
        pass


class FromLoadedSplicepointModel(nn.Module):
    motif_model_dict = True

    def __init__(self, splicepoint_model_path, num_motifs, splicepoint_index, **kwargs):
        super().__init__()
        self.model = load_individual_lssi_model(splicepoint_model_path, trainable=False)
        self.splicepoint_index = splicepoint_index
        assert num_motifs == 1

    def forward(self, x):
        motifs = self.model(x).log_softmax(-1)
        motifs = motifs[:, :, [self.splicepoint_index]]
        return dict(motifs=motifs)

    def notify_sparsity(self, sparsity):
        pass


class WithAndWithoutAdjustedDonor(ModularSplicePredictor):
    def __init__(self, *args, ad_index, ad_indicator_index, **kwargs):
        super().__init__(*args, **kwargs)
        self.ad_index = ad_index
        self.ad_indicator_index = ad_indicator_index

    def forward_post_motifs(
        self,
        post_sparse,
        splicepoint_results_residual,
        *,
        collect_intermediates,
        collect_losses,
        input_dict=None,
    ):
        setting = input_dict["setting"][:, 0, 0]

        assert (post_sparse[:, :, self.ad_indicator_index] == 0).all()

        # set the indicator to 1 if the setting is 1 (i.e. if the donor is being used)
        post_sparse[:, :, self.ad_indicator_index] = setting[:, None]
        # set the motif to 0 if the setting is 0 (i.e. if the donor is not being used)
        post_sparse[setting == 0, :, self.ad_index] = 0

        return super().forward_post_motifs(
            post_sparse,
            splicepoint_results_residual,
            collect_intermediates=collect_intermediates,
            collect_losses=collect_losses,
            input_dict=input_dict,
        )
