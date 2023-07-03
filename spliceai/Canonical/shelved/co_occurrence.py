from types import SimpleNamespace
from matplotlib import pyplot as plt
import pandas as pd
import tqdm.auto as tqdm
from permacache import permacache, stable_hash

import numpy as np

from shelved.extract_psams import extract_psams_from_motifs
from modular_splicing.data_for_experiments.standardized_sample import (
    standardized_sample,
)
from modular_splicing.motif_perturbations.perturbations_on_standardized_sample import (
    all_mpi_on_standardized_sample,
)
from modular_splicing.models_for_testing.list import FM
from modular_splicing.utils.plots.plot_psam import render_psams
from .extract_psams import extract_psams
from modular_splicing.utils.intron_exon_annotation import ANNOTATION_EXON

MOTIF_LABELS = "First Motif", "Second Motif"


@permacache(
    "validation/co_occurence/counts_on_binarized_motifs",
    key_function=dict(motifs_binarized=stable_hash, is_exon=stable_hash),
    multiprocess_safe=True,
)
def counts_on_binarized_motifs(motifs_binarized, is_exon, *, width):
    motifs_binarized = blur_bool_array(motifs_binarized, width)
    masks = {
        "exon": is_exon,
        "intron": ~is_exon,
        "all": np.ones_like(is_exon),
    }
    cl = motifs_binarized.shape[1] - is_exon.shape[1]
    motifs_binarized = motifs_binarized[
        :, cl // 2 : motifs_binarized.shape[1] - cl // 2, :
    ]
    print("done blurring")
    return {
        name: binary_counts_table(motifs_binarized & masks[name][:, :, None])
        for name in masks
    }


def iou(counts):
    return counts[1, 1] / (counts[1, 1] + counts[1, 0] + counts[0, 1])


def corr(counts):
    """
    r_xy = cov(x, y) / sqrt(var(x) var(y))

    cov(x, y) = E[XY] - E[X]E[Y]
              = pxy - px * py

    var(x) = E[X^2] - E[X]^2
            = px - px^2
    """
    count = sum(counts.values())
    freq = {k: counts[k] / count for k in counts}

    pxy = freq[1, 1]
    px = freq[1, 0] + freq[1, 1]
    py = freq[0, 1] + freq[1, 1]

    cov_xy = pxy - px * py
    var_x = px - px**2
    var_y = py - py**2

    return cov_xy / (var_x * var_y) ** 0.5


def mean_across(vals):
    if isinstance(vals[0], list):
        for val in vals:
            assert len(val) == len(vals[0])
        return [mean_across(x) for x in zip(*vals)]
    if isinstance(vals[0], dict):
        for val in vals:
            assert val.keys() == vals[0].keys()
        return {k: mean_across([val[k] for val in vals]) for k in vals[0]}
    assert isinstance(vals[0], np.ndarray)
    return np.mean(vals, axis=0)


def binary_counts_table(mot):
    mot = mot.reshape(-1, mot.shape[-1]).T
    universe = mot.shape[1]
    mot = [set(np.where(x)[0]) for x in tqdm.tqdm(mot, desc="where")]
    nm = len(mot)
    counts = {}
    chunks = np.array(
        [
            [
                [len(mot[a] & mot[b]), len(mot[a] - mot[b]), len(mot[b] - mot[a])]
                for a in range(nm)
            ]
            for b in tqdm.trange(nm, desc="intersection")
        ]
    )
    counts[1, 1], counts[1, 0], counts[0, 1] = chunks.transpose(2, 0, 1)
    union = counts[1, 1] + counts[1, 0] + counts[0, 1]
    counts[0, 0] = universe - union
    return counts


def pairs_table_for_metric(filt_counts, table_names, names):
    return pairs_table(
        filt_counts,
        names,
        [
            f"{k}/{name}" if k is not None else name
            for k in filt_counts
            for name in table_names
        ],
        [corr, iou],
        ["corr", "I/U"],
    )


def pairs_table(filt_counts, names, table_names, metrics, metric_names):
    i_s, j_s = np.meshgrid(np.arange(len(names)), np.arange(len(names)))
    i_s, j_s = i_s[i_s < j_s], j_s[i_s < j_s]
    rows = [*np.array(names)[np.array([i_s, j_s])]]
    index = [*MOTIF_LABELS]
    for metric, metric_name in zip(metrics, metric_names):
        rows += [
            c[i_s, j_s]
            for c in [metric(eff) for k in filt_counts for eff in filt_counts[k]]
        ]
        index += [f"{metric_name} [{n}]" for n in table_names]

    frame = pd.DataFrame(rows, index=index).T
    frame.sort_values(f"{metric_names[0]} [{table_names[0]}]")[::-1]

    maximal_values = frame[[x for x in frame if x.startswith(metric_names[0])]].T.max()
    frame = frame.loc[maximal_values.sort_values()[::-1].index].copy()
    return frame


def expand_effect(pert_data, *, cl, sl, num_motifs):
    effect = np.zeros((len(pert_data), cl + sl, num_motifs))
    for i, pert in enumerate(pert_data):
        overall_effect = np.abs(pert.perturbed_pred - pert.pred).sum(-1)
        effect[i, pert.motif_pos, pert.motif_ids] = overall_effect
    return effect


@permacache(
    "validation/co_occurence/filtered_counts",
    key_function=dict(pert_data=stable_hash, is_exon=stable_hash),
    multiprocess_safe=True,
)
def filtered_counts(pert_data, *, is_exon, params, **kwargs):
    effect = expand_effect(pert_data, **kwargs)
    filtered = []
    for param in tqdm.tqdm(params):
        print(param)
        filtered.append(
            counts_on_binarized_motifs(
                effect > param["bar"], is_exon, width=param["width"]
            )
        )
    return filtered


def draw_top_psams(summary, fm_pert_data, names, xs, *, top_several=40, **kwargs):
    pairs_amounts = np.array(
        summary[[x for x in summary if x.startswith("I/U")]].to_numpy().tolist()
    )
    mask = ((-pairs_amounts).argsort(0).argsort(0) < top_several).any(1)
    comparisons = summary.loc[mask]
    draw_psam_comparisons(comparisons, fm_pert_data, names, xs, **kwargs)


def draw_psam_comparisons(
    comparisons, fm_pert_data, names, xs, *, cl, sl, path, prefixes=None, title=None
):
    if prefixes is None:
        prefixes = [""] * comparisons.shape[0]
    ms = expand_effect(fm_pert_data, cl=cl, sl=sl, num_motifs=len(names)) > 0
    n1s, n2s = comparisons["First Motif"], comparisons["Second Motif"]
    m1, m2 = [ms[:, :, [names.index(n) for n in ns]] for ns in (n1s, n2s)]
    p, idxs = psam_comparisons(
        prefixes=prefixes,
        xs=xs,
        n1s=n1s,
        n2s=n2s,
        m1=m1,
        m2=m2,
        mcl=21,
    )
    render_psams(p, ncols=4, names=idxs)
    if title is not None:
        plt.suptitle(title)
    plt.savefig(path, facecolor="white")
    plt.close()


def get_at_rich(w=11, amount=1000, to_binary=True):
    psams = extract_psams(
        FM.binarized_model(1).model,
        data_path="dataset_train_all.h5",
        mcl=20,
        count=amount,
        bs=32,
        pbar=tqdm.tqdm,
    )
    pad = (psams.shape[1] - w) // 2
    psams = psams[:, pad : psams.shape[1] - pad]
    at_rich = psams.mean(1)[:, [0, -1]].sum(-1)
    median = np.median(at_rich)
    plt.hist(at_rich * 100, bins=20)
    plt.axvline(median * 100, color="black", label=f"Median: {median:.0%}")
    plt.xlabel("AT richness (mean % A or T in psam)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
    if not to_binary:
        return at_rich
    return at_rich > median


def produce_pair_masks(table, names, at_rich, *, exclude_across):
    f, s = ([names.index(x) for x in table[k]] for k in MOTIF_LABELS)
    f_at, s_at = at_rich[f], at_rich[s]
    masks = {
        "at/at": f_at & s_at,
        "gc/gc": ~f_at & ~s_at,
        "at/gc": f_at ^ s_at,
    }
    if exclude_across:
        del masks["at/gc"]
    return f_at, s_at, masks


def co_occurrence_scatter_plot(
    summary, names, at_rich, ax, kx, ky, exclude_across=True
):
    all_values = []
    _, _, masks = produce_pair_masks(
        summary, names, at_rich, exclude_across=exclude_across
    )
    for name in masks:
        xs, ys = summary[kx][masks[name]] * 100, summary[ky][masks[name]] * 100
        ax.scatter(xs, ys, alpha=0.5, marker=".", s=2, label=name)
        all_values += [*xs, *ys]
    rang = np.percentile(all_values, 1), np.percentile(all_values, 99)
    ax.plot(rang, rang, color="black", linestyle="--")
    ax.set_xlabel(kx)
    ax.set_ylabel(ky)
    ax.set_xlim(*rang)
    ax.set_ylim(*rang)
    ax.legend()


def co_occurrence_ratio_histogram(summary, names, at_rich, ax, kx, ky, interval=0.1):
    _, _, masks = produce_pair_masks(summary, names, at_rich, exclude_across=True)
    xs, ys = summary[kx], summary[ky]
    ratio = np.log(np.array(ys / xs, dtype=np.float)) / np.log(2)
    kx, ky = kx.lstrip("I/U "), ky.lstrip("I/U ")
    ratio_label = f"log$_2$ ratio: {ky}/{kx}"
    mi, ma = np.percentile(ratio[np.isfinite(ratio)], [1, 99])
    bins = np.arange(mi - interval, ma + interval, interval)
    for name in masks:
        ax.hist(ratio[masks[name]], label=name, bins=bins, histtype="step")

    ax.set_xlabel(ratio_label)
    ax.set_ylabel("frequency")
    ax.legend()


def produce_filtered_counts(names, segments, models_to_check, *, amount, cl, sl):
    xs, annots = standardized_sample(
        "dataset_intron_exon_annotations_train_all.h5",
        cl=cl,
        amount=amount,
        sl=sl,
        datapoint_extractor_spec=dict(type="BasicDatapointExtractor", run_argmax=False),
    )
    all_pert_data = all_mpi_on_standardized_sample(
        models_to_check, always_use_fm=True, is_binary=False, sl=sl
    )
    params = [
        dict(bar=bar, width=width) for width in [11, 21, 51] for bar in [0, 0.005]
    ]

    table_names = [
        (f"effect > {param['bar']:.1%}" if param["bar"] > 0 else "all")
        + (f"; w={param['width']}")
        + (f"; {segment}")
        for param in params
        for segment in segments
    ]
    filt_counts = {}
    for k in tqdm.tqdm(all_pert_data):
        print(k)
        counts = filtered_counts(
            all_pert_data[k],
            is_exon=annots[:, :, 0] == ANNOTATION_EXON,
            params=params,
            cl=cl,
            sl=sl,
            num_motifs=len(names),
        )
        filt_counts[k] = counts
    filt_counts = {
        k: [x[segment] for x in filt_counts[k] for segment in segments]
        for k in filt_counts
    }
    return SimpleNamespace(filt_counts=filt_counts, table_names=table_names, xs=xs)


def blur_bool_array(x, w):
    padding_amount = [(0, 0)] * len(x.shape)
    padding_amount[1] = (w // 2, w // 2)
    padded = np.pad(x, padding_amount)
    return np.any(
        [padded[:, i : padded.shape[1] - (w - 1 - i)] for i in range(w)], axis=0
    )


def psam_comparisons(*, prefixes, xs, n1s, n2s, m1, m2, mcl):
    """
    Produce a comparison of psams between two sets of motifs
    """

    assert len(prefixes) == len(n1s) == len(n2s)

    idxs = [
        f"{prefix}{source}"
        for prefix, n1, n2 in zip(prefixes, n1s, n2s)
        for source in [n1, n2, f"{n1} \\ {n2}", f"{n2} \\ {n1}"]
    ]
    first = extract_psams_from_motifs(xs, m1, mcl)
    second = extract_psams_from_motifs(xs, m2, mcl)
    diff1 = extract_psams_from_motifs(xs, m1 & ~m2, mcl)
    diff2 = extract_psams_from_motifs(xs, ~m1 & m2, mcl)

    p = np.array([first, second, diff1, diff2]).transpose((1, 0, 2, 3))
    p = p.reshape(-1, *p.shape[2:])
    return p, idxs
