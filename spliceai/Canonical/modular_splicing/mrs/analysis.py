import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

from .motifs import mean_subset_table_results


def produce_mean_difference_statistics(data, am, seeds, names):
    """
    Produce mean difference statistics for the given data, for
        donor, acceptor, and the mean of the two.

    See produce_summaries_from for the meaning of the keys in the returned
    dictionary.
    """
    mean_don = mean_subset_table_results(data[5]["y"], am, data[5]["x"], seeds, names)
    mean_acc = mean_subset_table_results(data[3]["y"], am, data[3]["x"], seeds, names)

    means = {"donor": mean_don, "acceptor": mean_acc}

    values = {}
    for k in means:
        values.update(produce_summaries_from(means[k], k))

    for k in ("bma", "aag"):
        unnormalized_donor = values[k, "donor"][0]
        unnormalized_acceptor = values[k, "acceptor"][0]

        normalized_donor = unnormalized_donor / values["bma", "donor"][0].std()
        normalized_acceptor = unnormalized_acceptor / values["bma", "acceptor"][0].std()

        values[k, "mean_normalized"] = (
            (normalized_donor + normalized_acceptor) / 2,
            values[k, "donor"][1],
        )

        values[k, "mean"] = (
            (unnormalized_donor + unnormalized_acceptor) / 2,
            values[k, "donor"][1],
        )

    return values


def produce_summaries_from(mean, tag):
    """
    Produce BMA and AAG summaries.
    """
    values = {
        "bma": (
            mean["FM"] - mean["~FM"],
            r"Baseline Motif Activity (BMA)",
        ),
        "aag": (
            mean[r"AM \ FM"] - mean[r"FM \ AM"],
            r"AM-FM Difference (AFD)",
        ),
    }

    return {(k, tag): values[k] for k in values}


def plot_against(names, values, x, y, *, title_prefix="", ax=None):
    """
    Plot the given keys against each other.

    Arguments:
        names: the names of each motif
        values: Full dictionary of all values
        x: The key to use for the x axis
        y: The key to use for the y axis
        title_prefix: A prefix to add to the title
        ax: The axis to plot on. If None, a new figure is created.

    Returns:
        the p value for the sign correlation.
    """
    highlight = {"TRA2A", "HNRNPA1", "HNRNPA2B1"}
    if ax is None:
        ax = plt.gca()
    xs, xl = values[x]
    ys, yl = values[y]

    ordering = sorted(range(len(xs)), key=lambda i: xs.index[i] in highlight)
    ax.scatter(
        xs[ordering],
        ys[ordering],
        color=["#44f" if x in highlight else "#aaa" for x in xs.index[ordering]],
    )
    for name in names:
        if name in highlight:
            ax.annotate(xy=(xs[name], ys[name]), text=name, size=10)
    ax.axvline(0, color="black")
    ax.axhline(0, color="black")
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    mi, ma = np.nanmin([xs, ys]), np.nanmax([xs, ys])
    mi, ma = mi * 1.2, ma * 1.2
    ax.set_xlim(mi, ma)
    ax.set_ylim(mi, ma)
    r, p = scipy.stats.pearsonr(xs[xs + ys == xs + ys], ys[xs + ys == xs + ys])
    ax.set_title(
        title_prefix
        + f"Correlation: r={r:.2f} p={p:.2e}"
        + "\n"
        + sign_alignment_summary(xs, ys)
    )


def sign_alignment_summary(xs, ys):
    """
    Summarize the alignment of the signs of the given series.
    Produces a percentage of the signs that are aligned, along with the
        p value for the sign correlation.
    """
    sign_correct = (xs >= 0) == (ys >= 0)
    chi_square = scipy.stats.chisquare([sign_correct.sum(), (~sign_correct).sum()])
    return f"Sign Agreement: {sign_correct.mean():.0%} p={chi_square.pvalue:.2e}"


def paper_figure_analysis(names, data, am, seeds, axs=None):
    """
    Do the analysis necessary to produce the figure in the paper.

    Arguments:
        names: the names of each motif
        data: Full dictionary of all data. See load.py for the format.
        am: The table of all models
        seeds: The seeds used in the experiment
    """
    values = produce_mean_difference_statistics(data, am, seeds, names)
    if axs is None:
        _, axs = plt.subplots(
            1, 2, figsize=(8, 4), dpi=200, facecolor="white", tight_layout=True
        )
    for i, k in enumerate(["donor", "acceptor"]):
        plot_against(
            names,
            values,
            ("bma", k),
            ("aag", k),
            title_prefix=f"For {k}\n",
            ax=axs[i],
        )

    return dict(
        unnormalized=sign_alignment_summary(
            values[("bma", "mean")][0], values[("aag", "mean")][0]
        ),
        normalized=sign_alignment_summary(
            values[("bma", "mean_normalized")][0], values[("aag", "mean_normalized")][0]
        ),
    )
