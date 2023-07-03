from matplotlib import pyplot as plt
import numpy as np


def get_windows_around_sites(ms, ys, mcl):
    categorized = {1: [], 2: []}
    for m, y in zip(ms, ys):
        dcl = m.shape[0] - y.shape[0]
        assert mcl - dcl >= 0
        delta = (mcl - dcl) // 2
        y = y[delta : y.shape[0] - delta]
        for y_idx in np.where(y)[0]:
            categorized[y[y_idx]].append(m[y_idx : y_idx + mcl + 1])
    categorized = {k: np.concatenate(categorized[k], axis=1) * 100 for k in categorized}
    return categorized


def meta_exon_plot(ms, ys, axs, *, mcl=600, samples=500, window, name="{}"):
    categorized = get_windows_around_sites(ms, ys, mcl=mcl)
    xs = np.arange(-mcl // 2, mcl // 2 + 1)
    assert len(axs) == len(categorized)
    for ax, cat in zip(axs, categorized):
        ax.plot(
            xs,
            categorized[cat][
                :,
                np.random.choice(
                    categorized[cat].shape[1], replace=False, size=samples
                ),
            ],
            alpha=3 / samples,
            marker=".",
            linestyle=" ",
            color="black",
        )
        ax.plot(
            xs,
            np.percentile(categorized[cat], 50, axis=1),
            label="Median",
            color="green",
        )
        ax.fill_between(
            xs,
            *[np.percentile(categorized[cat], p, axis=1) for p in (25, 75)],
            label="25th-75th %ile",
            alpha=0.2,
            color="green"
        )
        ax.plot(xs, categorized[cat].mean(1), label="Mean", color="black")
        ax.set_xlim(-window, window)
        ax.set_ylabel("Pairing %")
        ax.set_xlabel("Offset from site")
        ax.set_title({1: name.format("3'"), 2: name.format("5'")}[cat])
    ax.legend()
    return {k: categorized[k][(-window <= xs) & (xs <= window)] for k in categorized}


def binarize(yps, ys, low_multiplier, high_multiplier):
    remove = (yps.shape[1] - ys.shape[1]) // 2
    yps = yps[:, remove : yps.shape[1] - remove]
    threshs_low, threshs_high = [
        [
            np.quantile(yps[:, :, c], 1 - mult * (ys == c + 1).mean())
            for c in range(yps.shape[-1])
        ]
        for mult in (low_multiplier, high_multiplier)
    ]
    yps = (yps >= threshs_low) & (yps <= threshs_high)
    yps = yps.any(-1) * (yps.argmax(-1) + 1)
    return yps
