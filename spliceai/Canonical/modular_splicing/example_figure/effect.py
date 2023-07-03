import numpy as np

from modular_splicing.motif_perturbations.compute_perturbations import (
    motif_perturbations_individual,
)


def compute_effect(
    model,
    exon,
    effect_thresh,
    cl,
    *,
    kind_of_effect="linear",
    minimal_probability_requirement_for_effect,
    always_include_at_least=5,
):
    """
    Compute the effects for the given exon, using the given model.

    Returns a list of effects.

    Presented as a generator of dictionaries, with keys:
        - mot_pos: the position of the motif (in y-space coordinates)
        - mot_idx: the index of the motif
        - feat_pos: the position of the feature (in y-space coordinates)
        - feat_idx: the index of the feature (0 = acceptor, 1 = donor)
        - log_eff: the log effect size (log(without knockdown / with knockdown))

    The effect must be greater than effect_thresh, which is computed as per the
        kind of effect. If kind_of_effect is "linear", then effect_thresh is
        compared to the absolute difference in probabilites, and if
        kind_of_effect is "log", then effect_thresh is compared to the log
        difference in probabilities.

    The effect must also be on a feature with probability greater than
        minimal_probability_requirement_for_effect, or that is a true
        splice site.
    """
    pert = motif_perturbations_individual(
        model, exon["x"], exon["y"], threshold_info=None, include_threshold=np.exp(-10)
    )
    assert (pert.perturbed_motif_vals == 0).all()
    # negated because these are all knockdowns

    mask = (pert.motif_pos >= cl // 2) & (pert.motif_pos < exon["y"].shape[0] + cl // 2)

    perturbed_pred = pert.perturbed_pred[mask]
    pred = pert.pred
    motif_pos = pert.motif_pos[mask]
    motif_ids = pert.motif_ids[mask]

    effects = -(perturbed_pred - pred)
    log_effects = -(np.log(perturbed_pred) - np.log(pred))

    probs = np.array([np.exp(exon[k]["res"]) for k in ["FM", "AM"]]).max(0)
    # include sites that meet the threshold
    effect_mask = probs[pert.relevant_idxs] > minimal_probability_requirement_for_effect
    # include real sites
    effect_mask |= np.eye(3, dtype=np.bool)[exon["y"]][:, 1:][pert.relevant_idxs]

    if kind_of_effect == "linear":
        effect_of_interest = effects
    elif kind_of_effect == "log":
        effect_of_interest = log_effects
    else:
        raise ValueError(f"Unknown kind of effect: {kind_of_effect}")
    effect_of_interest = np.abs(effect_of_interest)
    above_thresh = effect_of_interest > effect_thresh

    relevant_effects = sorted(effect_of_interest[:, effect_mask].flatten())[
        -always_include_at_least:
    ]

    top_k = effect_of_interest >= (
        relevant_effects[0] if relevant_effects else -float("inf")
    )

    mot_idxs, feat_idxs = np.where((above_thresh | top_k) & effect_mask)

    for mot_idx, feat_idx in zip(mot_idxs, feat_idxs):
        yield dict(
            mot_pos=motif_pos[mot_idx] - cl // 2,
            mot_id=motif_ids[mot_idx],
            feat_pos=pert.relevant_idxs[0][feat_idx],
            feat_id=pert.relevant_idxs[1][feat_idx],
            log_eff=log_effects[mot_idx, feat_idx],
            above_thresh=above_thresh[mot_idx, feat_idx],
        )
