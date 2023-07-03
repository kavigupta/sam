from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt

from permacache import permacache, stable_hash

from modular_splicing.utils.construct import construct


def plot_rocs_with_and_or(
    motifs,
    donor_strengths,
    true_donors,
    donorlikes,
    *,
    pbar,
    thresholds=np.linspace(-30, 10, 1000),
    title=None,
    ax=None,
):
    by_idx, roc_default = get_all_rocs(
        motifs, donor_strengths, true_donors, donorlikes, thresholds, pbar=pbar
    )
    labels = set()

    def plot_roc(fpr, tpr, label=None, **kwargs):
        if label in labels:
            label = None
        labels.add(label)
        ax.plot(np.array(fpr) * 100, np.array(tpr) * 100, label=label, **kwargs)

    if ax is None:
        plt.figure(dpi=200)
        ax = plt.gca()
    for idx in donorlikes:
        plot_roc(
            *by_idx[idx].roc_and, color="red", alpha=0.5, label="LM & (d > thresh)"
        )
        plot_roc(
            *by_idx[idx].roc_or, color="green", alpha=0.5, label="LM | (d > thresh)"
        )
    plot_roc(*roc_default, color="black", label="d > thresh")
    ax.set_xlim(0, 10)
    ax.set_ylim(70, 100)
    ax.set_xlabel("FPR [%]")
    ax.set_ylabel("TPR [%]")
    if title is not None:
        ax.set_title(title)
    ax.legend()
    return by_idx


@permacache(
    "renderer/get_all_rocs",
    key_function=dict(
        motifs=stable_hash,
        donor_strengths=stable_hash,
        true_donors=stable_hash,
        donorlikes=stable_hash,
        thresholds=stable_hash,
        pbar=None,
    ),
)
def get_all_rocs(motifs, donor_strengths, true_donors, donorlikes, thresholds, *, pbar):
    by_idx = {}
    for idx in pbar(donorlikes):
        lm = motifs[:, 200:-200, idx]
        by_idx[idx] = best_shift(true_donors, lm)

    roc_default = roc(
        spec=dict(type="i"),
        thresholds=thresholds,
        donor_strengths=donor_strengths,
        true_donors=true_donors,
    )
    for idx in pbar(donorlikes):
        by_idx[idx].roc_and = roc(
            spec=dict(type="a", x=by_idx[idx].lm_shifted),
            thresholds=thresholds,
            donor_strengths=donor_strengths,
            true_donors=true_donors,
        )
        by_idx[idx].roc_or = roc(
            spec=dict(type="o", x=by_idx[idx].lm_shifted),
            thresholds=thresholds,
            donor_strengths=donor_strengths,
            true_donors=true_donors,
        )

    return by_idx, roc_default


@permacache(
    "analysis/best_shift", key_function=dict(true_donors=stable_hash, lm=stable_hash)
)
def best_shift(true_donors, lm):
    shifts = np.arange(-5, 5 + 1)
    best_shift = shifts[
        np.argmax([(np.roll(lm, shift, 1) & true_donors).sum() for shift in shifts])
    ]
    lm_shifted = np.roll(lm, best_shift, 1)
    get_shift = SimpleNamespace(lm=lm, best_shift=best_shift, lm_shifted=lm_shifted)

    return get_shift


@permacache(
    "renderer/roc",
    key_function=dict(
        thresholds=stable_hash, donor_strengths=stable_hash, true_donors=stable_hash
    ),
)
def roc(spec, thresholds, donor_strengths, true_donors):
    tpr = []
    fpr = []
    for t in thresholds:
        predictions = construct(
            dict(
                i=lambda left: left,
                a=lambda left, x: left & x,
                o=lambda left, x: left | x,
            ),
            spec,
            left=donor_strengths > t,
        )
        tpr.append(predictions[true_donors].mean())
        fpr.append(predictions[~true_donors].mean())
    return fpr, tpr
