import numpy as np

from modular_splicing.data_for_experiments.standardized_sample import (
    eclip_motifs_on_standardized_sample,
    standardized_sample,
)
from modular_splicing.utils.intron_exon_annotation import (
    ANNOTATION_EXON,
    ANNOTATION_INTRON,
)
from .eclip_peaks import eclips_from_onehot


def extract_actual_range(eclip, mode):
    if mode[0] == "range":
        return eclip["start"], eclip["end"]
    elif mode[0] == "from_5'":
        return eclip["end"] - mode[1], eclip["end"] + mode[1]
    else:
        raise RuntimeError(f"Invalid mode: {mode[0]}")


def exon_intron_masks(annotation_path, data_amount):
    """
    Uses the standard data order, produces masks of exons and introns

    Args:
        annotation_path: The path to the annotation file.
        data_amount: The amount of data to use.

    Returns a dictionary with the keys "all", "exon", "intron"
        each key corresponds to a different mask
        each value has shape (data_amount, SL), where SL is the sequence length of the referenced file
    """

    _, annotations = standardized_sample(
        annotation_path,
        amount=data_amount,
        datapoint_extractor_spec=dict(type="BasicDatapointExtractor", run_argmax=False),
    )
    annotations = annotations[:, :, 0]
    return dict(
        all=np.ones_like(annotations, dtype=np.bool),
        introns=annotations == ANNOTATION_INTRON,
        exons=annotations == ANNOTATION_EXON,
    )


def get_eclips_for_testing(eclip_motifs_array, mode, masks):
    """
    Prepare eclip data for testing by extracting eclip peaks as objects, producing control eclips,
    and then filtering them by category.

    Args:
        eclip_motifs_array: The eclip motifs array. Of shape (data_amount, SL, *eclip_shape)
        mode: The mode that will be used when using the eclips
        masks: A dictionary of masks. See exon_intron_masks.

    Returns:
         A dictionary of eclips. The keys are tuples of (category, "control" or "real")
    """
    eclips_for_motif = eclips_from_onehot(eclip_motifs_array)
    control_eclips = create_control_eclips(eclips_for_motif, *masks["all"].shape)
    return {
        (name, "control" if is_control else "real"): filter_eclips(
            control_eclips if is_control else eclips_for_motif,
            masks[name],
            mode=mode,
        )
        for name in masks
        for is_control in (0, 1)
    }


class EclipDataForTesting:
    """
    Produce eclip data for testing

    Args:
        amount: The amount of data to use
        eclip_idxs: The eclip indices to use
        mode: The mode that will be used when using the eclips
        path: The path to the data file
        annotation_path: The path to the exon/intron annotation file

    Returns an EclipDataForTesting object with parameters
        all_eclip_motifs: The eclip motifs array. Of shape (data_amount, SL, *eclip_shape)
        filtered_eclips: A dictionary of eclips. The keys are tuples of (category, "control" or "real")
    """

    def __init__(self, amount, eclip_idxs, *, mode, path, annotation_path):
        self.all_eclip_motifs = eclip_motifs_on_standardized_sample(
            path=path,
            amount=amount,
            eclip_params=dict(
                replicates_to_use=["1", "2"],
                one_hot_encoding_mode="eclip_original",
                dataset_pipeline_mode="with_spliceai_pipeline",
            ),
            eclip_idxs=eclip_idxs,
        )
        masks = exon_intron_masks(annotation_path, amount)
        self.filtered_eclips = get_eclips_for_testing(
            self.all_eclip_motifs, mode, masks
        )


def create_control_eclips(eclips, batches, seq_len):
    """
    Create a set of control eclips from the given set of eclips.

    The control eclips are created by randomly selecting a batch and start and end
        position for each eclip uniformally at random from the space of all possible
        positions.

    Args:
        eclips: The eclips to create controls for. List.
        batches: The number of batches in the data.
        seq_len: The sequence length of the data.

    Returns:
        A list of control eclips of equal length to `eclips`. Each control
            has the same motif index and replicate index as the original.
    """
    rng = np.random.RandomState(0)
    result = []
    for eclip in eclips:
        batch_idx = rng.choice(batches)
        length = eclip["end"] - eclip["start"]
        start = rng.randint(0, seq_len - length)
        end = start + length
        result.append(
            dict(
                batch_idx=batch_idx,
                motif_idx=eclip["motif_idx"],
                replicate_idx=eclip["replicate_idx"],
                start=start,
                end=end,
            )
        )
    return result


def filter_eclips(eclips, mask, *, mode):
    """
    Filter the given eclip list by the given mask and mode.

    Only eclips for whom the whole range is in the mask will be kept.

    Args:
        eclips: The eclips to filter. List.
        mask: The mask to filter by. Of shape (batches, seq_len).
        mode: The mode that will be used when using the eclips

    Returns:
        A sublist of the original list.
    """
    result = []
    for eclip in eclips:
        start, end = extract_actual_range(eclip, mode=mode)
        if mask[eclip["batch_idx"], start:end].all():
            result.append(eclip)
    return result
