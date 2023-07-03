from itertools import cycle

import attr

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_sensitivity_by_motif(full_data, names):
    """
    Plot sensitivity by motif, ordered by the mean.
    """
    sensitivity = {m: full_data[m].sum((0, 1)) for m in full_data}
    sensitivity = pd.DataFrame(sensitivity, index=names)
    sensitivity = sensitivity.loc[sensitivity.T.mean().sort_values().index[::-1]].copy()
    sensitivity = pd.concat(
        [pd.DataFrame(sensitivity.mean(), columns=["Mean"]).T, sensitivity]
    )
    plt.figure(dpi=120, figsize=(20, 8))
    colors = list(plt.rcParams["axes.prop_cycle"].by_key()["color"] * 2)[::-1]
    for i, col in enumerate(sensitivity):
        plt.scatter(
            np.arange(len(sensitivity))
            + np.random.RandomState(i).randn(len(sensitivity)) * 0.1,
            sensitivity[col],
            marker="o",
            facecolors="none",
            edgecolors=colors.pop(),
            # markersize=10,
            alpha=1,
            label=col,
        )
    plt.xticks(np.arange(len(sensitivity)), sensitivity.index, rotation=90)
    plt.grid()
    plt.legend()
    plt.show()
    return sensitivity


@attr.s
class EffectPlotter:
    """
    Class including code to plot the effect of a given mutation on the
    activity of the given targets.

    Structured a bit oddly because it used to be used to plot grouped clusters.
    Can probably be cleaned up further at this point.
    """

    full_data = attr.ib()
    names = attr.ib()
    width_each = attr.ib(default=5)
    height_each = attr.ib(default=5)
    limit_mode = attr.ib(default=lambda x: np.percentile(x, 99))

    def setup_plots(self, num_plots, unit="% of max", **kwargs):
        _, axs = plt.subplots(
            num_plots,
            2,
            figsize=(self.width_each * 2, self.height_each * num_plots),
            **kwargs,
        )
        if num_plots == 1:
            axs = [axs]
        for ax1, ax2 in axs:
            self.set_axes_plots(ax1, ax2, unit=unit)
        return axs

    def set_axes_plots(self, ax1, ax2, *, unit):
        for target, ax in zip("AD", (ax1, ax2)):
            ax.set_ylabel(f"Effect on {target} [{unit}]")
            ax.set_xlabel(f"Displacement from {target} [nt]")
            ax.grid()

    def plot_series_by_index(self, ax1, ax2, idx, style_kwargs, *, use_label=True):
        if isinstance(use_label, str):
            label = use_label
        else:
            label = self.names[idx]
        kwargs = dict(**style_kwargs)
        kwargs_label = dict(**kwargs, label=label) if use_label else kwargs

        radius = self.full_data.shape[1] // 2
        xvals = np.arange(-radius, radius + 1)

        ax1.plot(xvals, self.full_data[0, :, idx] * 100, **kwargs_label)
        ax1.plot(xvals, -self.full_data[1, :, idx] * 100, **kwargs)
        ax2.plot(xvals, self.full_data[2, :, idx] * 100, **kwargs_label)
        ax2.plot(xvals, -self.full_data[3, :, idx] * 100, **kwargs)

    def plot_for_indices(
        self,
        indices,
        *,
        name,
        legend=True,
        unit="% of max",
        figure_kwargs={},
        **kwargs,
    ):
        [[ax1, ax2]] = self.setup_plots(1, unit=unit, **figure_kwargs)

        for idx, color in zip(
            indices, cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
        ):
            self.plot_series_by_index(ax1, ax2, idx, dict(color=color, **kwargs))
        rv = self.limit_mode(self.full_data[:, :, list(indices)]) * 100
        if rv == 0:
            rv = 100
        for ax in ax1, ax2:
            ax.set_ylim(-rv, rv)

        if legend:
            ax2.legend()
        for ax in ax1, ax2:
            ax.set_title(name)
