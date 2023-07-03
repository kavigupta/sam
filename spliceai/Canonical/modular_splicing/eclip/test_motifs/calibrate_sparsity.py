import numpy as np

from permacache import permacache
from modular_splicing.legacy.hash_model import hash_model


def calibrate_sparsities(binarized_to_calibrate_to, non_binarized):
    """
    Returns a version of non_binarized with the same sparsity as binarized_to_calibrate_to,
        by magnitude.
    """
    _, thresholds = compute_sparsity_thresholds(
        binarized_to_calibrate_to, non_binarized
    )
    result = non_binarized > thresholds
    return result


def compute_sparsity_thresholds(binarized_to_calibrate_to, non_binarized):
    """
    Compute the sparsity thresholds necessary to calibrate the nonbinarized motifs to the corresponding binarized ones
    """
    non_binarized_sparsity = (non_binarized > 0).mean((0, 1))
    binarized_sparsity = binarized_to_calibrate_to.mean((0, 1))
    assert (non_binarized_sparsity > binarized_sparsity).all()
    thresholds = [
        np.quantile(
            non_binarized[:, :, motif_idx],
            1 - binarized_sparsity[motif_idx],
        )
        for motif_idx in range(binarized_to_calibrate_to.shape[-1])
    ]

    return binarized_sparsity, thresholds


@permacache(
    "eclip_analysis/test_motifs/compute_sparsity_thresholds_for_model_3",
    key_function=dict(
        binarized_to_calibrate_to_model=hash_model, non_binarized_model=hash_model
    ),
)
def compute_sparsity_thresholds_for_model(
    binarized_to_calibrate_to_model,
    non_binarized_model,
    *,
    path,
    motif_indices,
    amount,
    cl,
):
    """
    Compute sparsity thresholds for the given model, using the standard data.

    See compute_sparsity_thresholds for more details.
    """
    from modular_splicing.data_for_experiments.standardized_sample import (
        model_motifs_on_standardized_sample,
    )

    non_binarized = model_motifs_on_standardized_sample(
        model_for_motifs=non_binarized_model,
        indices=motif_indices,
        path=path,
        amount=amount,
        cl=cl,
    )
    binarized_to_calibrate_to = model_motifs_on_standardized_sample(
        model_for_motifs=binarized_to_calibrate_to_model,
        indices=motif_indices,
        path=path,
        amount=amount,
        cl=cl,
    )
    return compute_sparsity_thresholds(binarized_to_calibrate_to, non_binarized)
