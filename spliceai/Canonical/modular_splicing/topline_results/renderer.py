import numpy as np
import matplotlib.pyplot as plt

from modular_splicing.evaluation import standard_e2e_eval


def plot_topline_results(all_accuracies, ax):
    """
    Plot topline results as a bar chart with error bars.
    """
    all_accuracies = {
        k: [np.array(v) * 100 for v in vs] for k, vs in all_accuracies.items()
    }

    names = list(all_accuracies)

    for channel in range(2):
        accs_each = [np.array(all_accuracies[n])[:, channel] for n in names]
        means = [x.mean() for x in accs_each]
        mins, maxs = [x.min() for x in accs_each], [x.max() for x in accs_each]
        mins = np.array(mins)
        maxs = np.array(maxs)
        lens = np.array([len(x) for x in accs_each])

        xs = np.arange(len(names)) + (channel - 0.5) * 0.3

        ax.bar(
            xs,
            means,
            label=["3'", "5'"][channel],
            width=0.3,
            color=["#3f8733", "#a4d99c"][channel],
        )
        ax.errorbar(
            xs,
            (mins + maxs) / 2,
            yerr=(maxs - mins) / 2 + np.where(lens == 1, np.nan, 0),
            color="black",
            linestyle=" ",
            capsize=3,
        )
        for i in range(len(means)):
            ax.text(
                xs[i],
                mins[i] - 1,
                f"{means[i]:.1f}",
                ha="center",
                va="top",
                size=10,
                rotation=90,
            )

    ax.set_xticks(np.arange(len(all_accuracies)), names)
    ax.set_ylabel("Accuracy [%]")
    ax.set_ylim(0, 100)
    ax.legend()
