import copy

import torch
import torch.nn as nn

from modular_splicing.utils.construct import construct

from modular_splicing.models.motif_models.types import motif_model_types


class AlternateDonorSeparatingMotifModel(nn.Module):
    motif_model_dict = True
    _input_is_dictionary = True

    def __init__(
        self,
        underlying_motif_model_spec,
        original_donor_spec,
        donor_sparsity_multiplier,
        num_motifs,
        suppression_offsets,
        mask_above_threshold,
        shift_donor_amount,
        **kwargs,
    ):
        super().__init__()
        from modular_splicing.models.modules.lssi_in_model import BothLSSIModels

        self.model_to_adjust = construct(
            motif_model_types(),
            underlying_motif_model_spec,
            num_motifs=num_motifs - 1,
            **kwargs,
        )
        self.splicepoint_model = construct(
            dict(BothLSSIModels=BothLSSIModels), original_donor_spec
        )
        self.donor_sparsity_multiplier = donor_sparsity_multiplier
        self.suppression_offsets = suppression_offsets
        self.mask_above_threshold = mask_above_threshold
        self.shift_donor_amount = shift_donor_amount

    def forward(self, x):
        if isinstance(x, dict):
            x = x["x"]
        donors, padded_mask = self.get_donors(x)
        motifs_output = self.model_to_adjust(x)

        if not getattr(self.model_to_adjust, "motif_model_dict", False):
            motifs_output = dict(motifs=motifs_output)

        motifs = motifs_output["motifs"]

        motifs_output["padded_mask"] = padded_mask

        motifs = motifs * (~padded_mask[:, :, None]).float()
        motifs = torch.cat([motifs, donors[:, :, None]], axis=2)
        motifs_output["motifs"] = motifs
        return motifs_output

    def get_donors(self, x):
        from modular_splicing.models.modules.lssi_in_model import LSSI_MODEL_THRESH

        splicepoints = self.splicepoint_model.forward_just_splicepoints(x)
        donors = splicepoints[:, :, -1] - LSSI_MODEL_THRESH
        donors[donors < 0] = 0

        return self.shift_right(donors, self.shift_donor_amount), self.pad_mask(
            donors > self.mask_above_threshold
        )

    def shift_right(self, donors, shift_amount):
        return nn.functional.pad(donors, (shift_amount, 0))[:, : donors.shape[1]]

    def pad_mask(self, donor_mask):
        indices = [-x for x in self.suppression_offsets]

        pad_left = min([0] + indices)
        pad_right = max([0] + indices)

        donor_mask_padded = nn.functional.pad(
            donor_mask, (-pad_left, pad_right), "constant", 0
        )

        all_padded = [
            donor_mask_padded[:, idx - pad_left : idx - pad_left + donor_mask.shape[1]]
            for idx in indices
        ]
        all_padded = torch.stack(all_padded)
        all_padded = all_padded.any(0)
        return all_padded

    def notify_sparsity(self, sparsity):
        self.underlying_motif_model.notify_sparsity(sparsity)
        self.splicepoint_model.notify_sparsity(sparsity)


def AdjustedAlternateDonorSeparatingMotifModel(
    am_spec, alternate_donor_separating_spec, num_motifs, **kwargs
):
    am_spec = copy.deepcopy(am_spec)
    am_spec["sparsity_enforcer_spec"]["sparse_spec"] = dict(
        type="ParallelSpatiallySparse",
        sparse_specs=[
            am_spec["sparsity_enforcer_spec"]["sparse_spec"],
            dict(type="NoSparsity"),
        ],
        num_channels_each=[num_motifs - 1, 1],
        update_indices=[0],
        get_index=0,
    )

    assert not "model_to_adjust_spec" in am_spec
    return construct(
        motif_model_types(),
        am_spec,
        model_to_adjust_spec=alternate_donor_separating_spec,
        num_motifs=num_motifs,
        **kwargs,
    )
