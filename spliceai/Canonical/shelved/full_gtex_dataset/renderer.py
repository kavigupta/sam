from types import SimpleNamespace
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import tqdm.auto as tqdm

from permacache import permacache, stable_hash, drop_if_equal

from .compute_psi import all_psi_values


def directional_strand_histogram(all_table):
    plt.figure(dpi=200)
    bins = np.arange(-10, 25, 0.1)
    don_diff = all_table.don - all_table.don_other
    acc_diff = all_table.acc - all_table.acc_other
    mean_diff = (don_diff + acc_diff) / 2
    for d, l in (don_diff, "5'"), (acc_diff, "3'"), (mean_diff, "mean"):
        plt.hist(
            d,
            bins=bins,
            alpha=0.5,
            label=f"{l} diff [wrong direction: {(d < 0).mean():.2%}]",
        )
    plt.xlabel("difference of log spm scores (selected strand - other strand)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


@permacache(
    "dataset/gtex/renderer/num_tissue_types",
    dict(
        spm=stable_hash,
        min_cluster_size=drop_if_equal(0),
        max_cluster_size=drop_if_equal(float("inf")),
    ),
)
def num_tissue_types(
    spm, dataset_root, fasta_path, ws, min_cluster_size=0, max_cluster_size=float("inf")
):
    _, psi_values = all_psi_values(
        spm,
        dataset_root,
        fasta_path,
        ws=ws,
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
    )
    num_tissue_types = np.array([len(x) for x in psi_values])
    return num_tissue_types


def number_tissue_type_histogram(num_tissue_types, tables, ax):
    ax.hist(num_tissue_types, bins=np.arange(1 + len(tables)))
    ax.axvline(2, label=f"in at least 2: {np.sum(num_tissue_types >= 2)}", color="red")
    ax.axvline(
        49 / 2,
        label=f"at least half: {np.sum(num_tissue_types > 49/2)}",
        color="orange",
    )
    ax.axvline(48, label=f"in all: {np.sum(num_tissue_types == 49)}", color="blue")
    ax.set_xlabel("Number of tissue types the splicepoint is found in")
    ax.set_ylabel("Frequency")
    ax.legend()


def number_tissue_type_histograms(spm, tables, *, dataset_root, fasta_path, ws):
    max_cluster_sizes = [float("inf"), 3]

    nrows, ncols = 2, 1
    assert len(max_cluster_sizes) == nrows * ncols

    _, axs = plt.subplots(
        nrows, ncols, dpi=200, figsize=(ncols * 5, nrows * 3), tight_layout=True
    )
    axs = axs.flatten()
    for mcs, ax in zip(max_cluster_sizes, axs):
        ntt = num_tissue_types(
            spm, dataset_root, fasta_path, ws=ws, max_cluster_size=mcs
        )
        number_tissue_type_histogram(
            ntt,
            tables,
            ax,
        )
        ax.set_title(f"max cluster size: {mcs}; total count={len(ntt)}")


@permacache("dataset/gtex/renderer/psi_values_stats_by_width", dict(spm=stable_hash))
def psi_values_stats_by_width(spm, dataset_root, fasta_path, ws):
    _, psi_values = all_psi_values(spm, dataset_root, fasta_path, ws=ws)
    psi_values_all = [
        [x[w] for xs in psi_values for x in xs.values()] for w in tqdm.tqdm(ws)
    ]
    means = [np.mean(x) for x in psi_values_all]
    upper_quartile = [np.percentile(x, 75) for x in psi_values_all]
    lower_quartile = [np.percentile(x, 25) for x in psi_values_all]
    sample_idxs = np.random.RandomState(0).choice(len(psi_values_all[0]), size=1000)
    samples = [[x[i] for i in sample_idxs] for x in psi_values_all]
    return SimpleNamespace(
        means=means,
        upper_quartile=upper_quartile,
        lower_quartile=lower_quartile,
        samples=samples,
    )


def plot_psi_values_stats_by_width(spm, dataset_root, fasta_path, ws):
    sbw = psi_values_stats_by_width(spm, dataset_root, fasta_path, ws)
    samples = np.array(sbw.samples)

    _, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=200, tight_layout=True)

    ax = axs[0]
    ax.plot(ws, np.array(sbw.means) * 100, label="Mean")
    ax.fill_between(
        ws,
        np.array(sbw.lower_quartile) * 100,
        np.array(sbw.upper_quartile) * 100,
        alpha=0.25,
        label="25th to 75th",
    )
    ax.set_ylabel("psi [%]")
    ax.legend()
    ax.set_title("Aggregated statistics")
    ax.grid()

    ax = axs[1]
    ax.plot(ws, 100 * samples, color="black", alpha=0.1)
    ax.set_title("Individual trajectories")
    ax.set_ylabel("psi [%]")

    ax = axs[2]
    ax.plot(ws, 100 * (samples < 0.1).mean(-1), label="psi < 0.1")
    ax.plot(
        ws,
        100 * ((0.1 <= samples) & (samples <= 0.9)).mean(-1),
        label="0.1 <= psi <= 0.9",
    )
    ax.plot(ws, 100 * (0.9 < samples).mean(-1), label="0.9 < psi")
    ax.legend()
    ax.grid()
    ax.set_ylabel("% of splicepoints with given psi")
    ax.set_title("By psi category")

    for ax in axs:
        ax.set_xlabel("Width")
        ax.set_xscale("log")
        ax.set_xticks(ws, rotation=90)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
