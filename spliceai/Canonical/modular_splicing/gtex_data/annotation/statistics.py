from collections import defaultdict
from matplotlib import pyplot as plt

from permacache import permacache, stable_hash
import pandas as pd

from modular_splicing.gtex_data.annotation.compute_optimal_sequence import (
    compute_optimal_sequences_all,
)

bin_boundaries = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, float("inf")]


@permacache(
    "modular_splicing/gtex_data/annotation/statistics/compute_counts_by_type",
    key_function=dict(ensg_names=stable_hash),
)
def compute_counts_by_type(ensg_names, cost_params):
    counts_each = defaultdict(int)
    by_length_class = defaultdict(lambda: defaultdict(int))
    seq_all = compute_optimal_sequences_all(ensg_names, cost_params=cost_params)
    for k in seq_all:
        intermediates, _, annotations_chosen = seq_all[k]
        length = len(intermediates["sites"])
        for x in annotations_chosen:
            by_length_class[length][type(x).__name__] += len(x.sites)
        counts_each[length] += 1
    return dict(counts_each.items()), {
        k: dict(v.items()) for k, v in by_length_class.items()
    }


def length_proj(x):
    for start, end in zip(bin_boundaries, bin_boundaries[1:]):
        if start <= x < end:
            return start
    else:
        assert not "reachable"


def bin_name(x):
    i = bin_boundaries.index(x)
    return f"{bin_boundaries[i]}-{bin_boundaries[i + 1] - 1}"


def compute_binned_counts(counts_each, by_length_class):
    counts_by_bin = defaultdict(int)
    for length, count in counts_each.items():
        counts_by_bin[length_proj(length)] += count
    by_bin = defaultdict(lambda: defaultdict(int))
    for length, by_type in by_length_class.items():
        for t, count in by_type.items():
            by_bin[length_proj(length)][t] += count

    by_bin = pd.DataFrame(by_bin).fillna(0)
    by_bin = by_bin[sorted(by_bin)]
    by_bin = by_bin / by_bin.sum(0)
    by_bin = by_bin.rename(columns={x: bin_name(x) for x in by_bin})
    return dict(counts_by_bin.items()), by_bin


def display_binned_counts(by_length_class):
    _, axs = plt.subplots(1, 2, figsize=(10, 5))
    for ax, lim in zip(axs, [100, 5]):
        for x in by_length_class.index:
            ax.plot(by_length_class.loc[x] * 100, label=x)
        ax.set_xticks(range(by_length_class.shape[1]))
        ax.set_xticklabels(list(by_length_class), rotation=90)
        ax.set_xlabel("Splicepoints per gene")
        ax.set_ylabel("Percentage by class")
        ax.legend()
        ax.set_ylim(0, lim)
        ax.grid()


def display_counts_by_type(counts_each):
    plt.plot(range(len(counts_each)), [counts_each[k] for k in sorted(counts_each)])
    plt.xticks(range(len(counts_each)), [bin_name(x) for x in sorted(counts_each)])
    plt.xticks(rotation=90)
    plt.xlabel("Splicepoints per gene")
    plt.ylabel("Number of genes")
    plt.show()
