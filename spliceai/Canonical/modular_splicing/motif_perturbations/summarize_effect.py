from types import SimpleNamespace
import numpy as np


def positional_effects_near_splicepoints(
    data,
    *,
    num_motifs,
    blur_radius,
    effect_radius=100,
    normalize_mode,
    splicepoint_types="AD",
    **kwargs,
):
    """
    Positional effects of motifs near splicepoints. This is a wrapper around
    `motif_prevalence_and_effect_near_splicepoints` that computes the positional effects
    for all motifs and all splicepoints.

    Parameters
    ----------
    data : list
        List of perturbation data (output of `motif_perturbations_individual`) on
        several sampled sequences.
    num_motifs : int
        Number of motifs in the data.
    blur_radius : int
        Radius of the blur to apply to the positional effects.
    effect_radius : int
        Radius of the window to compute the indices and effects for.
    normalize_mode : one of ["by_motifs_presence", "by_total_effect"]
        How to normalize the positional effects.
        If "by_motifs_presence", normalize each position to the overall presence of the motif.
            Effectively, divide each motif's effect at a given position class by the number of
            features that have the motif at any position class. This allows us to see at which
            positions the motif is most effective.
        If "by_total_effect", normalize each position to the total effect of the all motifs.
            Effectively, divide each motif's effect at a given position class by the total effect
            of all motifs at that position class. This allows us to compare motif to motif at each
            position.
    splicepoint_types : str
        Which splicepoint types to include. "A" for acceptor, "D" for donor, "B" for branch site.
        Concatenate these into a string.
    **kwargs
        Additional arguments to `motif_prevalence_and_effect_near_splicepoints`.

    Returns
    -------
    A table of effects, of shape (2 * len(splicepoint_types), 2 * effect_radius + 1, num_motifs)
    where the first axis is
        splicepoint type 0 [+ effect], splicepoint type 0 [- effect]
        splicepoint type 1 [+ effect], splicepoint type 1 [- effect]
        ...
    the second axis is the position, and the third axis is the motif.
    """
    results = []
    presence = []
    for splicepoint_type in splicepoint_types:
        source = motif_prevalence_and_effect_near_splicepoints(
            data,
            outcome=splicepoint_type,
            actual=True,
            num_motifs=num_motifs,
            cl=400,
            radius=effect_radius,
            **kwargs,
        )
        results += [source.pos_effect, source.neg_effect]
        presence += [source.presence] * 2
    transformer = lambda rs: np.array(
        [avg_blur(np.nan_to_num(xs).T, blur_radius) for xs in rs]
    )
    full_data = transformer(results)
    presence = transformer(presence)
    if normalize_mode == "by_motifs_presence":
        full_data = full_data / (presence.sum(1)[:, None] + 1e-10)
    elif normalize_mode == "by_total_effect":
        full_data = full_data / (full_data.sum() + 1e-10)
    else:
        raise RuntimeError(f"Invalid normalization mode: {normalize_mode}")
    return full_data


def motif_prevalence_and_effect_near_splicepoints(
    perturb_motifs,
    outcome,
    actual,
    num_motifs,
    cl,
    sl=1000,
    radius=100,
    minimal_unconfidence=0.1,
):
    """
    Compute the prevalence and effect of motifs near splicepoints.

    Parameters
    ----------
    perturb_motifs : list
        List of perturbation data (output of `motif_perturbations_individual`) on
        several sampled sequences.
    outcome: one of ["A", "D", "B"]
        Only compute prevalence and effect for features of this type. "B" indicates a branch site
            and is only relevant when a motif predicts branch sites
    actual : bool
        If True, compute the prevalence and effect of features that are real (TP/FN). If False,
            compute the prevalence and effect of features that are fake (FP/TN).
    num_motifs : int
        Number of motifs in the data.
    cl : int
        Context length.
    sl: int
        Sequence length.
    radius : int
        Radius of the window to compute the indices and effects for.
    minimal_unconfidence : float
        Only compute prevalence and effect for features whose predictions are between `minimal_unconfidence`
            and `1 - minimal_unconfidence`. This is to avoid including features where our model
            is very confident.

    Returns
    -------
    three arrays, all of shape (num_motifs, 2 * radius + 1)
    where the first axis is the motif index and the second axis is the position
        prevalence: Prevalence of the motif at the given position (i.e., how many features
            have the motif at the given position).
        positive_effect: Positive effect of the motif at the given position. (i.e., how much
            does the feature's prediction increase when the motif is activated, among sites
            where it does increase).
        negative_effect: Negative effect of the motif at the given position. (i.e., how much
            does the feature's prediction decrease when the motif is activated, among sites
            where it does decrease).
    """

    def plausible_preds(d):
        return (minimal_unconfidence < d.pred) & (d.pred < 1 - minimal_unconfidence)

    presence = np.zeros((num_motifs, 2 * radius + 1))
    pos_effect = np.zeros((num_motifs, 2 * radius + 1))
    neg_effect = np.zeros((num_motifs, 2 * radius + 1))
    count = 0
    for pm in perturb_motifs:
        feat_mask = (
            plausible_preds(pm)
            & (pm.actual == actual)
            & (pm.relevant_idxs[1] == {"A": 0, "D": 1, "B": 2}[outcome])
        )
        for feat in np.where(feat_mask)[0]:
            index, deltas = compute_nearby_indices_and_effects(
                pm, feat_idx=feat, radius=radius, sl=sl, cl=cl
            )
            if index is None:
                continue
            presence[index] += 1
            pos_effect[index] += positive_effect(deltas)
            neg_effect[index] += negative_effect(deltas)
            count += 1
    pos_effect /= count
    neg_effect /= count
    presence /= count
    return SimpleNamespace(
        presence=presence,
        pos_effect=pos_effect,
        neg_effect=neg_effect,
    )


def compute_nearby_indices_and_effects(pm, *, feat_idx, radius, sl, cl):
    """
    Compute the indices and effects for nearby motifs of a given feature.

    If the feature is more than radius away from the edge of the data we have,
        return None, None.

    Parameters
    ----------
    pm : SimpleNamespace
        Perturbation data for a single feature. Output of `motif_perturbations_individual`.
    feat_idx : int
        Index of the feature to compute the nearby indices and effects for. Acts as an
            index into the `pm`'s list of features.
    radius : int
        Radius of the window to compute the indices and effects for.
    sl : int
        Sequence length.
    cl : int
        Context length.

    Returns
    -------
    index : tuple (motif_ids, position), each of size (K,)
        motif_ids: Indices of the motifs.
        position: Positions of the motifs. In relative-to-feature
            space from (0, 2 * radius) where
            being on top of the feature puts you at position `radius`.
    deltas : np.ndarray (K,)
        Effects of the motifs on the given feature. Treated as if the
            motif is always activated. I.e., if a motif being knocked out
            has a positive effect, the effect is negative.
    """
    # convert the feature index into a location in x-space.
    centerpos = pm.relevant_idxs[0][feat_idx] + cl // 2
    if not (0 <= centerpos - radius and centerpos + radius < sl + cl):
        # too close to the edge of the sequence to compute
        return None, None

    # motif centers relative to the center of the feature
    shifted_motif_centers = pm.motif_pos - centerpos
    relevant_motifs_mask = np.abs(shifted_motif_centers) <= radius
    index = (
        pm.motif_ids[relevant_motifs_mask],
        # shifts from (-radius, radius) to (0, 2 * radius)
        shifted_motif_centers[relevant_motifs_mask] + radius,
    )
    # effects of the perturbations on the feature
    delta_pred = pm.perturbed_pred[relevant_motifs_mask, feat_idx] - pm.pred[feat_idx]
    # if the motif is knocked out, the effect should be flipped
    motif_directional_sign = np.sign(
        pm.perturbed_motif_vals[relevant_motifs_mask]
        - pm.motif_vals[relevant_motifs_mask]
    )
    deltas = delta_pred * motif_directional_sign
    return index, deltas


def positive_effect(x):
    """
    Return the positive part of x.
    """
    return x * (x > 0)


def negative_effect(x):
    """
    Return the negative part of x.
    """
    return -x * (x < 0)


def avg_blur(xs, radius):
    """
    Take the average of the values in xs with a window of size 2 * radius + 1.
    """
    return radial_sum(xs, radius) / radial_sum(np.ones_like(xs), radius)


def radial_sum(xs, radius):
    """
    Take the sum of the values in xs with a window of size 2 * radius + 1. At the
        edges, just sum the values that are in the window.
    """
    xs = np.array(xs)
    xs = np.concatenate(
        [np.zeros((radius, *xs.shape[1:])), xs, np.zeros((radius, *xs.shape[1:]))]
    )
    return np.array(
        [xs[off : xs.shape[0] - (2 * radius - off)] for off in range(2 * radius + 1)]
    ).sum(0)
