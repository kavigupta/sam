"""
Contains methods for determining the distances between motifs and their nearest neighbor.

Useful in analysis for the "motifs conflict with each other and that's why width is needed" theory.
"""

from permacache import permacache

import numpy as np
from modular_splicing.evaluation.predict_motifs import predict_motifs
from modular_splicing.data_for_experiments.standardized_sample import (
    standardized_sample,
)
from modular_splicing.legacy.hash_model import hash_model


def _gather_data_on_motif_distances(m, amount):
    """
    Gathers from the training dataset the distances betwen motifs
    along with other information

    Parameters
        m: the model to use
        amount: the amount of data to use; will use the standard sequence length

    Returns
        motif_idxs: the indices of the motifs that are collected
        distance_to_closest: the distance to the closest motif for each motif index
        num_motifs: the number of motifs (just a number)
    """
    xs, _ = standardized_sample("dataset_train_all.h5", amount, cl=0)
    motifs = predict_motifs(m, xs)[:, :, :-1] > 0
    batch_idxs, seq_idxs, motif_idxs = np.where(motifs)
    same_batch_mask = (batch_idxs[1:-1] == batch_idxs[:-2]) & (
        batch_idxs[1:-1] == batch_idxs[2:]
    )
    right_dist = seq_idxs[2:] - seq_idxs[1:-1]
    left_dist = seq_idxs[1:-1] - seq_idxs[:-2]
    distance_to_closest = np.minimum(right_dist, left_dist)

    motif_idxs = motif_idxs[1:-1][same_batch_mask]

    distance_to_closest = distance_to_closest[same_batch_mask]
    num_motifs = motifs.shape[-1]
    return motif_idxs, distance_to_closest, num_motifs


@permacache(
    "ms/width_experiments/distance_between_motifs/nearest_neighbor_motifs_differences_table",
    key_function=dict(m=hash_model),
)
def nearest_neighbor_motifs_differences_table(m, amount, max_d):
    """
    Return a table of the frequencies of distances for each motif

    Parameters
        m: the model to use
        amount: the amount of data to use; will use the standard sequence length
        max_d: the maximum distance to consider

    Returns
        by_distance
        by_distance[i][j] is the number of times motif j appears within i of the nearest neighbor
    """
    motif_idxs, distance_to_closest, num_motifs = _gather_data_on_motif_distances(
        m, amount
    )
    by_distance = np.zeros((max_d + 1, num_motifs), dtype=np.int)
    np.add.at(by_distance, (np.clip(distance_to_closest, 0, max_d), motif_idxs), 1)
    return by_distance


@permacache(
    "ms/width_experiments/distance_between_motifs/_nearest_neigboring_motifs_distances_sums_and_counts",
    key_function=dict(m=hash_model),
)
def _nearest_neigboring_motifs_distances_sums_and_counts(m, amount):
    """
    Returns the sum and count of each motif's distances to its nearest neighbor
    """
    motif_idxs, distance_to_closest, num_motifs = _gather_data_on_motif_distances(
        m, amount
    )

    sums, counts = np.zeros((2, num_motifs))
    np.add.at(sums, motif_idxs, distance_to_closest)
    np.add.at(counts, motif_idxs, 1)
    return sums, counts


def nearest_neigboring_motifs_distances(m, amount):
    """
    Returns the mean distance from each motif to its nearest neighbor
    """
    sums, counts = _nearest_neigboring_motifs_distances_sums_and_counts(m, amount)
    return sums / counts
