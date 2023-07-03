import numpy as np
import h5py
from modular_splicing.dataset.additional_data import AdditionalData

from modular_splicing.dataset.h5_dataset import H5Dataset


class UnifiedAlternativeDataset(H5Dataset):
    def __init__(self, *args, underlying_ordering, rewriters=(), **kwargs):
        super().__init__(
            *args,
            datapoint_extractor_spec=dict(
                type="BasicDatapointExtractor",
                run_argmax=False,
                clip_y_zeros=False,
                rewriters=[
                    dict(
                        type="AdditionalChannelDataRewriter",
                        combinator_spec=dict(type="ReplacingCombinator"),
                        out_channel=["outputs", "y"],
                        data_provider_spec=dict(
                            type="expand_unified_alternative_dataset"
                        ),
                    )
                ]
                + list(rewriters),
            ),
            **kwargs,
        )
        with h5py.File(self.path, "r") as f:
            ordering = [x.decode("utf-8") for x in f["ordering"][:]]
            assert (
                underlying_ordering == ordering
            ), f"{underlying_ordering} != {ordering}"


class NonConflictingAlternativeDataset(UnifiedAlternativeDataset):
    """
    Idea is to take a multi-outcome dataset and produce a single outcome dataset
    but with the positions other splicepoints are active on masked out, so that
    they don't get trained/tested on, either positively or negatively.
    """

    def __init__(
        self,
        *args,
        underlying_ordering,
        outcome_to_pick,
        channels_per_outcome,
        mask_channel_offsets,
        always_keep_picked=False,
        actually_mask_others=True,
        **kwargs,
    ):
        start_idx = underlying_ordering.index(outcome_to_pick) * channels_per_outcome
        relevant_channels = list(range(start_idx, start_idx + channels_per_outcome))

        rewriters = []
        if actually_mask_others:
            rewriters += [
                dict(
                    type="AdditionalChannelDataRewriter",
                    out_channel=["outputs", "mask"],
                    data_provider_spec=dict(
                        type="mask_out_other_splicepoints_dataset",
                        channels_per_outcome=channels_per_outcome,
                        chosen_channels=relevant_channels,
                        mask_channel_offsets=mask_channel_offsets,
                        always_keep_picked=always_keep_picked,
                    ),
                )
            ]

        rewriters += [
            dict(
                type="AdditionalChannelDataRewriter",
                out_channel=["outputs", "y"],
                data_provider_spec=dict(
                    type="select_channels", relevant_channels=relevant_channels
                ),
                combinator_spec=dict(type="ReplacingCombinator"),
            ),
        ]

        super().__init__(
            *args,
            underlying_ordering=underlying_ordering,
            rewriters=rewriters,
            **kwargs,
        )


class MaskOutOtherSplicepointsDatasetRewriter(AdditionalData):
    def __init__(
        self,
        channels_per_outcome,
        chosen_channels,
        mask_channel_offsets,
        always_keep_picked,
    ):
        self.channels_per_outcome = channels_per_outcome
        self.chosen_channels = chosen_channels
        self.mask_channel_offsets = mask_channel_offsets
        self.always_keep_picked = always_keep_picked

    def compute_additional_input(self, original_input, path, i, j):
        raise RuntimeError("Not implemented")

    def compute_additional_output(self, original_output, path, i, j, cl_max):
        out_channels = original_output.shape[-1]
        mask_channels = [
            x
            for x in range(out_channels)
            if x not in self.chosen_channels
            and x % self.channels_per_outcome in self.mask_channel_offsets
        ]
        result = ~original_output[..., mask_channels].any(-1)
        if self.always_keep_picked:
            always_keep_channels = [
                x
                for x in range(out_channels)
                if x in self.chosen_channels
                and x % self.channels_per_outcome in self.mask_channel_offsets
            ]
            result = result | (original_output[..., always_keep_channels].any(-1))
        return result


class SelectChannels(AdditionalData):
    def __init__(self, relevant_channels):
        self.relevant_channels = relevant_channels

    def compute_additional_input(self, original_input, path, i, j):
        raise RuntimeError("Not implemented")

    def compute_additional_output(self, original_output, path, i, j, cl_max):
        return original_output[..., self.relevant_channels]


class ExpandUnifiedAlternativeDatasetRewriter(AdditionalData):
    def compute_additional_input(self, original_input, path, i, j):
        raise RuntimeError("Not implemented")

    def compute_additional_output(self, original_output, path, i, j, cl_max):
        del path, i, j
        assert original_output.shape[-1] % 2 == 0
        alternatives = original_output.shape[-1] // 2

        for_each_alternative = np.array(
            [original_output[..., i * 2 : (i + 1) * 2] for i in range(alternatives)]
        )
        contains_data = for_each_alternative.sum(-1)

        assert 0 <= contains_data.min() <= contains_data.max() <= 1

        bottom_one_hot = 1 - contains_data[:, :, None]
        for_each_alternative = np.concatenate(
            [bottom_one_hot, for_each_alternative], axis=-1
        )
        result = np.concatenate(for_each_alternative, axis=-1)

        return result
