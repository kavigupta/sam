from types import SimpleNamespace

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from modular_splicing.motif_perturbations.summarize_effect import (
    positional_effects_near_splicepoints,
)


def effect(pert, num_motifs):
    full_data = positional_effects_near_splicepoints(
        pert,
        num_motifs=num_motifs,
        blur_radius=1,
        effect_radius=400,
        normalize_mode="by_motifs_presence",
    )
    return full_data


def donor_effect(eff, mcl=13):
    center = eff.shape[1] // 2
    return eff[2, center - mcl // 2 : center + mcl // 2 + 1].mean(0) / eff[
        2, center - 100 : center + 100 + 1
    ].mean(0)


def donor_overlap(donors, motifs, mwidth=13):
    cl = motifs.shape[1] - donors.shape[1]
    batch_idxs, seq_idxs = np.where(donors)
    batch_idxs, seq_idxs = (
        batch_idxs[None],
        seq_idxs[None] + np.arange(-(mwidth // 2), mwidth // 2 + 1)[:, None],
    )
    return motifs[batch_idxs, seq_idxs + cl // 2].any(0).mean(0)


DONOR_WINDOW_K = 43
DONOR_WINDOW_OFF = DONOR_WINDOW_K // 2
WINDOW_SPAN = np.arange(-DONOR_WINDOW_OFF, DONOR_WINDOW_OFF + 1)
STANDARD_DONOR = -2.5, 20.5


def extract_windows(donors, ms):
    assert DONOR_WINDOW_K % 2 == 1
    windows = []
    for batch_idx, seq_idx in zip(*np.where(donors)):
        if seq_idx < DONOR_WINDOW_OFF or seq_idx > ms.shape[1] - DONOR_WINDOW_OFF - 1:
            continue
        windows.append(
            ms[batch_idx, seq_idx - DONOR_WINDOW_OFF : seq_idx + DONOR_WINDOW_OFF + 1]
        )
    return np.array(windows)


def extract_all_windows(donors, xs, dmotifs):
    donor_windows = extract_windows(donors, xs)
    windows = {m: extract_windows(donors, dmotifs[m]) for m in dmotifs}
    standard_logo = donor_windows.mean(0)
    return SimpleNamespace(
        donor_windows=donor_windows, windows=windows, standard_logo=standard_logo
    )


def relative_binding_sites(m, donors, off):
    cl = m.shape[1] - donors.shape[1]
    batch_idx, seq_idx = np.where(donors)
    seq_idx += off + cl // 2
    return m[batch_idx, seq_idx]


def relative_binding_sites_correlation(motifs, offsets_by_motif, donors):
    sites = {}
    for mod, idx in offsets_by_motif:
        sites[mod, idx] = relative_binding_sites(
            motifs[mod]["motifs"][:, :, idx], donors, offsets_by_motif[mod, idx]
        )

    labels = list(offsets_by_motif.keys())
    result = []
    for k1 in sites:
        result.append([])
        for k2 in sites:
            result[-1].append(np.corrcoef(sites[k1], sites[k2])[0, 1])
    return pd.DataFrame(result, index=labels, columns=labels)


def plot_effects_vs_overlaps(data, motifs, num_motifs, effects, bar):
    lm_models = sorted(effects)

    donor_effects = {path: donor_effect(effects[path]) for path in lm_models}

    donor_overlaps = {
        path: donor_overlap(data.donors, motifs[path]["motifs"]) for path in lm_models
    }

    all_effects = np.concatenate([donor_effects[m] for m in lm_models])
    all_overlaps = np.concatenate([donor_overlaps[m] for m in lm_models])
    all_model_idxs = np.concatenate(
        [[i] * num_motifs[m] for i, m in enumerate(lm_models)]
    )
    plt.figure(dpi=120)
    plt.fill_between([70, 100], [0, 0], [bar, bar], color="red", alpha=0.1)
    plt.fill_between(
        [0, 100], [bar, bar], [all_effects.max() * 1.1] * 2, color="blue", alpha=0.1
    )
    plt.scatter(
        all_overlaps * 100, all_effects, c=all_model_idxs, cmap="jet", marker="."
    )
    plt.xlabel("% of donors covered by the motif")
    plt.ylabel("+ effect near donor / baseline")
    plt.title("Points colored by which model")
    # plt.yscale("log")
    plt.grid()
    return {model: donor_effects[model] > bar for model in lm_models}
