from permacache import permacache, stable_hash, drop_if_equal
import tqdm.auto as tqdm

import numpy as np
from modular_splicing.evaluation.predict_motifs import predict_motifs

from modular_splicing.utils.run_batched import run_batched


def is_kaway(arr, k):
    """
    Takes a boolean array (batch, sequence) and returns a boolean array of the same shape
    that is true if the given site is k away from a position in the original array that is true.
    """
    return (
        np.pad(arr, ((0, 0), (0, k)))[:, k:]
        | np.pad(arr, ((0, 0), (k, 0)))[:, : arr.shape[1]]
    )


def distance_to_feature_center(is_feature, maximum=100):
    """
    Returns the distance to the nearest feature center.

    If no feature is not found within maximum, returns maximum.
    """
    result = [is_feature]
    for k in tqdm.trange(1, maximum):
        if result[-1].all():
            break
        result.append(is_kaway(is_feature, k) | result[-1])
    result.append(np.ones_like(result[-1]))
    return np.argmax(result, 0)


@permacache(
    "modular_splicing/base_perturbations/swing_by_distance",
    key_function=dict(
        swings=stable_hash,
        validity_mask=stable_hash,
        dist=stable_hash,
        dist_divide=drop_if_equal(1),
    ),
)
def swings_by_distance(swings, validity_mask, dist, ks, dist_divide=1):
    """
    Get the mean swing values by distance category.

    Args:
        swings (N, L): The swing values associated with each position
        validity_mask (N, L): A boolean array indicating which positions are valid to be analyzed
        dist (N, L): The distance to the nearest feature center for each position
        ks (list): The distance categories to analyze

    Returns:
        means (list): The mean swing values for each distance category
        lows (list): The lower bound of the 95% confidence interval for each distance category
        highs (list): The upper bound of the 95% confidence interval for each distance category
    """
    means = []
    lows = []
    highs = []
    for k in tqdm.tqdm(ks):
        els = swings[validity_mask & (dist // dist_divide == k)] * 100
        bootstrap_means = (
            np.random.RandomState(0).choice(els, size=(1000, els.shape[0])).mean(1)
        )
        low, high = np.percentile(bootstrap_means, [2.5, 97.5])
        means += [np.mean(els)]
        lows += [low]
        highs += [high]
    return means, lows, highs


def plot_swings_by_distance(
    swings, validity_mask, dist, ks, *, color, label, ax, dist_divide
):
    means, lows, highs = swings_by_distance(
        swings, validity_mask, dist, ks, dist_divide
    )
    ax.plot(ks * dist_divide, means, label=label, color=color)
    ax.fill_between(ks * dist_divide, lows, highs, color=color, alpha=0.25)


def create_validity_mask(cl, avoid_range):
    validity_mask = np.arange(1 + cl)
    validity_mask = (validity_mask < cl // 2 - avoid_range) | (
        validity_mask > cl // 2 + avoid_range
    )
    return validity_mask


def plot_swings_by_distance_to_features(
    fm, xfulls, swings, *, mcl, model_cl, ax, dist_divide=1, max_value=20
):
    ks = np.arange(max_value // dist_divide)

    validity_mask = create_validity_mask(model_cl, avoid_range=40)

    dist = distance_to_feature_center(predict_motifs(fm, xfulls).any(-1))[
        :, mcl // 2 : -(mcl // 2)
    ]
    splice_sites = run_batched(
        fm.splicepoint_model.forward_just_splicepoints, xfulls, 1000, pbar=tqdm.tqdm
    )
    dist_splice = distance_to_feature_center((splice_sites >= -10).any(-1))[
        :, mcl // 2 : -(mcl // 2)
    ]
    plot_swings_by_distance(
        swings,
        validity_mask,
        dist_splice,
        ks,
        ax=ax,
        color="red",
        label="LSSI site",
        dist_divide=dist_divide,
    )
    plot_swings_by_distance(
        swings,
        validity_mask,
        dist,
        ks,
        ax=ax,
        color="blue",
        label="Motif Center",
        dist_divide=dist_divide,
    )
    plot_swings_by_distance(
        swings,
        validity_mask & (dist_splice > 20),
        dist,
        ks,
        ax=ax,
        color="cyan",
        label="Motif Center [away from LSSI]",
        dist_divide=dist_divide,
    )
    ax.set_xlabel("Distance from nearest feature")
    ax.grid()
