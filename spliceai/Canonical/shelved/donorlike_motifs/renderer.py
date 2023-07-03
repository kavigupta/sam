from itertools import cycle

import numpy as np
import matplotlib.pyplot as plt

from modular_splicing.utils.plots.plot_psam import render_psams

from .analysis import DONOR_WINDOW_OFF, WINDOW_SPAN, STANDARD_DONOR


def draw_windows(model, motif_idx, window):
    plt.figure(dpi=120)
    values = window.mean(0) * 100
    cpoints = sorted(
        np.where(values > 15)[0] - DONOR_WINDOW_OFF,
        key=lambda x: -values[x + DONOR_WINDOW_OFF],
    )

    plt.scatter(WINDOW_SPAN, values, color="black")
    plt.title(f"{model} LM{motif_idx + 1}")

    plt.axvspan(
        *STANDARD_DONOR,
        alpha=0.1,
        color="black",
        label="Donor model window",
    )
    colors = ["green", "blue", "red", "cyan", "purple", "orange", "yellow", "magenta"]
    for color, cpoint in zip(cycle(colors), cpoints):
        plt.fill_between(
            [cpoint - 4.5, cpoint + 4.5],
            [values[cpoint + DONOR_WINDOW_OFF] - 10],
            [values[cpoint + DONOR_WINDOW_OFF] + 10],
            alpha=0.1,
            color=color,
            label=f"Window for {cpoint}",
        )
        plt.scatter(
            WINDOW_SPAN[cpoint + DONOR_WINDOW_OFF],
            values[cpoint + DONOR_WINDOW_OFF],
            color=color,
        )
    plt.xlabel("Displacement from 5'")
    plt.ylabel("% of time motif appears")
    plt.legend()
    plt.grid()
    plt.ylim(0, 100)
    plt.show()


def display_matrix(results, models, ax, title):
    results = np.array(results)
    results[np.eye(results.shape[0], dtype=bool)] = np.nan
    im = ax.imshow(results, vmin=-0.2, vmax=0.7, cmap="jet")
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, rotation=90)
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels(models)
    ax.set_title(title)
    return im


def relative_to(windows, offsets_by_motif, *, motif_tuples, motif_names, baseline):
    diffs = np.array(
        [windows.donor_windows.mean(0)]
        + [
            windows.donor_windows[
                windows.windows[m][:, offsets_by_motif[m] + DONOR_WINDOW_OFF]
            ].mean(0)
            - windows.donor_windows.mean(0)
            for m in motif_tuples
        ]
    )
    names = [f"Baseline: {baseline}"] + motif_names

    axs = render_psams(
        diffs,
        names=names,
        psam_mode="raw",
        figure_kwargs=dict(dpi=120),
    )

    for ax in axs:
        ax.get_xaxis().set_visible(False)
