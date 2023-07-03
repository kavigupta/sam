import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict

from modular_splicing.utils.statistics import bootstrap


def plot_by_model_group(
    vals,
    names,
    title,
    ylabel,
    *,
    ax=None,
    colors,
    extra_zero_line=True,
    error_bars=True,
):
    """
    Run a plot by model group, with bootstrap confidence intervals across seeds.

    Scatter plot.

    Parameters:
        vals: list of values, parallel to names
        names: list of names, parallel to vals
        title: title of the plot
        ylabel: y-axis label
        ax: Axis to plot on. If None, create a new figure.
        colors: dict of colors for each model group
    """
    grouped_by_prefix = defaultdict(list)
    for name in names:
        *n, _ = name.split("_")
        grouped_by_prefix["_".join(n)].append(name)
    gnames = list(grouped_by_prefix)
    if ax is None:
        plt.figure(dpi=200)
        ax = plt.gca()
    for i, gname in enumerate(gnames):
        group = grouped_by_prefix[gname]
        elements = [vals[names.index(n)] for n in group]
        elements = np.array(elements)
        elements = elements[np.isfinite(elements)]
        if elements.size == 0:
            continue
        ax.scatter([i] * len(elements), elements, color=colors[gname], marker=".")
        ax.scatter([i], np.mean(elements), color=colors[gname])
        if error_bars:
            lo, hi = bootstrap(elements)
            ax.fill_between(
                [i - 0.25, i + 0.25], lo, hi, color=colors[gname], alpha=0.2
            )
    ax.grid()
    ax.set_xticks(np.arange(len(gnames)), gnames, rotation=90)
    ax.set_xlim(-1, len(gnames))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if extra_zero_line:
        ax.axhline(0, color="black")


def plot_grouped_results(results, *, colors={}, ax=None):
    """
    Plot the given grouped results. Results should be a map from group name to
    a map from seed to a list of values.

    Parameters:
        results: map from group name to map from seed to list of values
        colors: map from group name to color. If not specified, use black.
    """
    vals, names = [], []
    for name in results:
        for seed in results[name]:
            if seed == "path":
                continue
            names.append(f"{name}_{seed}")
            vals.append(results[name][seed])
    vals = np.array(vals) * 100

    plot_by_model_group(
        vals,
        names,
        "",
        "Accuracy [%]",
        colors={name: colors.get(name, "black") for name in results},
        extra_zero_line=False,
        error_bars=False,
        ax=ax,
    )
