from types import SimpleNamespace

from more_itertools import chunked

import numpy as np
from permacache import permacache, stable_hash
import torch
from modular_splicing.data_for_experiments.standardized_sample import (
    standardized_sample,
)


@permacache(
    "modular_splicing/motif_perturbations/motif_statistics/motif_statistics",
    key_function=dict(
        m=stable_hash,
        bs=None,
        pbar=None,
    ),
)
def motif_statistics(
    m,
    *,
    count,
    bs,
    min_value,
    max_value,
    resolution=0.001,
    sl=5000,
    pbar=lambda x, **kwargs: x,
):
    """
    Produce statistics on the motifs' distribution.
        For best performancee, set `min_value` and `max_value`
        to large values, and `resolution` to a small value.

    Arguments:
        m: The model to use.
        count: The number of sequences to use in the computation.
        bs: The batch size to use. Should not affect the results.
        min_value, max_value, resolution: parameters for how to
            discretize the motif scores.
        sl: The sequence length to use.
        pbar: A progress bar function. Should not affect the results.

    Returns: a simple namespace with keys
        original_thresholds: the thresholds the model uses to cut
            off the motifs.
        near_miss_thresholds: the thresholds that would have been
            used to cut off the motifs if the model has 2x sparsity.
            In essence, cut off for motifs that nearly were on but were
            turned off by the sparsity.
        mean_above_thresh: the mean of the motif scores above the
            threshold.
    """
    print("Computing motif statistics for ", stable_hash(m))
    threshs = m.sparsity_enforcer.thresholds_numpy
    num_motifs = threshs.shape[0]
    bin_vals, bins = compute_bins_for_motif_model(
        m,
        sl=sl,
        count=count,
        bs=bs,
        pbar=pbar,
        min_value=min_value,
        max_value=max_value,
        resolution=resolution,
    )
    empirical_sparsities, mean_above_thresh = compute_threshold_statistics(
        bins=bins,
        bin_vals=bin_vals,
        resolution=resolution,
        min_value=min_value,
        threshs=threshs,
    )
    near_miss_thresholds = get_near_miss_thresholds(
        min_value=min_value,
        resolution=resolution,
        num_motifs=num_motifs,
        bins=bins,
        empirical_sparsities=empirical_sparsities,
    )
    return SimpleNamespace(
        original_thresholds=threshs,
        near_miss_thresholds=np.array(near_miss_thresholds),
        mean_above_thresh=mean_above_thresh,
    )


def get_near_miss_thresholds(
    *, min_value, resolution, num_motifs, bins, empirical_sparsities
):
    """
    Compute the thresholds at which the sparsity is 2x the empirical sparsity.

    Arguments:
        min_value: The minimum value of the bins.
        resolution: The resolution of the bins.
        num_motifs: The number of motifs.
        bins: The bins.
        empirical_sparsities: The empirical sparsities.

    Returns:
        The near miss thresholds.
    """
    double_sparsities = empirical_sparsities * 2
    near_miss_thresholds = []
    for k in range(num_motifs):
        # total number of motifs seen * the target sparsity we want to hit
        # = the number of motifs we want to see
        target = int(bins[k].sum() * double_sparsities[k])
        bins_from_top = np.cumsum(bins[k, ::-1])
        # finds u such that bins_from_top[u-1] < target <= bins_from_top[u]
        u = np.searchsorted(bins_from_top, target)
        u = (bins.shape[1] - 1) - u
        # u such that
        # bins[k, u-1:].sum() = bins_from_top[::-1][u-1] < target <= bins_from_top[::-1][u] = bins[k, u:].sum()
        # we want to ensure that bins[k, u:].sum() ~= target, so this works directly
        near_miss_thresholds.append(u * resolution + min_value)
    return near_miss_thresholds


def compute_threshold_statistics(*, min_value, resolution, threshs, bins, bin_vals):
    """
    Computes the empirical sparsities for the motifs, as well as the mean value of the
        motif above the threshold, for the given motif values and bin values.

    Parameters
    ----------
    min_value, resolution: see compute_bins_for_motif_model
    threshs: the sparse thresholds for each motif
    bins, bin_vals: see compute_bins_for_motif_model

    Returns
    -------
    empirical_sparsities: the empirical sparsities for each motif, i.e., how much
        of the motif values are above the threshold
    mean_above_thresh: the mean value of the motif above the threshold
    """
    num_motifs = bins.shape[0]
    shifted_threshs = ((threshs - min_value) / resolution).round().astype(np.int)
    empirical_sparsities, mean_above_thresh = [], []
    for j in range(num_motifs):
        overall_count = bins[j].sum()
        # get the bins that are actually included (above threshold)
        above_thresh_bins = bins[j, shifted_threshs[j] :]
        # percentage of values above threshold
        empirical_sparsities.append(above_thresh_bins.sum() / overall_count.sum())
        # weighted average of the values above threshold
        mean_above_thresh.append(
            (bin_vals[shifted_threshs[j] :] * above_thresh_bins).sum()
            / above_thresh_bins.sum()
        )
    empirical_sparsities, mean_above_thresh = np.array(empirical_sparsities), np.array(
        mean_above_thresh
    )

    return empirical_sparsities, mean_above_thresh


def compute_bins_for_motif_model(
    m, *, sl, count, bs, pbar, min_value, max_value, resolution
):
    """
    Compute bins for the motif model. Do so by iterating through the first
    `count` sequences of length `sl` and computing the motif scores for each
    position. Then produce a histogram of the motif scores.

    Arguments:
        m: the motif model
        sl: the sequence length
        count: the number of sequences to use
        bs: the batch size
        pbar: a progress bar function
        min_value: the minimum value to consider
        max_value: the maximum value to consider
        resolution: the resolution of the values to consider

    Returns:
        bin_vals: the values of the bins: (B,)
        bins: the bin counts: (M, B) of how many motif scores
            fall into each bin for each motif
    """
    num_motifs = m.sparsity_enforcer.thresholds_numpy.shape[0]

    xs, _ = standardized_sample("dataset_train_all.h5", count, cl=m.cl)

    # one bin for every value in the range [min_value, max_value] at
    # a step size of `resolution`
    bins = np.zeros(
        (num_motifs, int((max_value - min_value) / resolution)), dtype=np.int
    )
    # map from the motif index in the sequence of relevant motifs to
    # motif index in the model's matrix output
    motif_index = {v: k for k, v in m.sparsity_enforcer.motif_index().items()}
    for xs in pbar(chunked(xs, bs), total=(count + bs - 1) // bs):
        xs = np.array(xs)
        with torch.no_grad():
            m.eval()
            motifs = (
                m(torch.tensor(xs).cuda(), only_motifs=True)["nonsparse_motifs"]
                .cpu()
                .numpy()
            )
        # collapse batch/sequence dimensions
        motifs = motifs.reshape(-1, motifs.shape[-1]).T
        # rescale to match the bins
        motifs = ((motifs - min_value) / resolution).round().astype(np.int)
        # clip to our range
        motifs = np.clip(motifs, 0, bins.shape[1] - 1)
        for j in range(num_motifs):
            mot_id = motif_index[j]
            # add to the histogram
            np.add.at(bins[j], motifs[mot_id], 1)
    # rescale the values back to the original shape
    bin_vals = np.arange(bins.shape[1]) * resolution + min_value
    return bin_vals, bins
