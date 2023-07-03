import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import logomaker


def render_psam(psam, psam_mode="normalized", ax=None):
    """
    Render a PSAM as a LogoMaker logo. Any NaNs in the PSAM will be replaced with near-0 values
        so that they don't cause the LogoMaker to crash. This means that if you provide an
        all-NaN PSAM, you'll get a uniform logo

    Parameters
    ----------
    psam : np.ndarray
        The PSAM to render. Shape (L, 4).
    psam_mode : str
        One of "raw", "normalized", or "info".
            - "raw" means to plot the raw PSAM
            - "normalized" means to first ensure that the PSAM sums to 1
                at each position, and then plot the result.
            - "info" means to treat the PSAM as a probability distribution
                over nucleotides at each position, and then plot the
                `sequence logo` by plotting a PSAM that sums to 2 - H(p)
                at each position.
    ax : matplotlib.Axis
        The axis to plot on. If None, use the current axis.
    """
    psam = np.array(psam)
    assert len(psam.shape) == 2 and psam.shape[1] == 4, str(psam.shape)
    assert psam_mode in {"raw", "normalized", "info"}

    if ax is None:
        ax = plt.gca()
    if psam_mode == "info":
        psam = information_psam(psam)
    if psam_mode == "normalized":
        psam = psam / psam.sum(1)[:, None]

    psam = np.nan_to_num(psam, 0.001)
    psam[(psam == 0).all(1)] = 0.001

    psam_df = pd.DataFrame(psam, columns=list("ACGT"))
    logo = logomaker.Logo(psam_df, ax=ax)
    return logo


def information_psam(psam):
    psam = np.array(psam)
    psam = psam / psam.sum(-1)[..., None]
    entropies = (-psam * np.log(psam + 1e-30) / np.log(2)).sum(-1)[..., None]
    informations = 2 - entropies
    psam = psam * informations
    return psam


def render_psams(
    psams,
    *,
    names,
    ncols=1,
    width=5,
    figure_kwargs=dict(),
    axes_mode="completely_blank",
    same_ylim=False,
    **kwargs,
):
    """
    Render several PSAMs in a grid. The grid will have `ncols` columns, and as many rows
        as necessary to fit all the PSAMs.

    Each PSAM will be rendered using `render_psam`, and the `kwargs` will be passed to it.
        The canvas for each PSAM will be a `width`-inch wide by `1.6`-inch tall figure.

    Parameters
    ----------
    psams : list of np.ndarray
        The PSAMs to render. Each PSAM should be a 2D array of shape (L, 4).
    names : list of str
        The names of the PSAMs. These will be used as the titles of the subplots.
    ncols : int
        The number of columns in the grid.
    width : float
        The width of each subplot, in inches.
    figure_kwargs : dict
        The keyword arguments to pass to `plt.figure`.
    axes_mode : str
        One of "completely_blank" or "just_y". If "completely_blank", then the axis will
            be completely blank, with no ticks or labels. If "just_y", then the axis will
            be blank except for the y-axis.
    same_ylim : bool
        If True, then all the subplots will have the same y-axis limits.
    **kwargs
        The keyword arguments to pass to `render_psam`.
    """
    assert len(names) == len(psams)
    grid_size = [(len(psams) + ncols - 1) // ncols, ncols]
    _, axs = plt.subplots(
        *grid_size, figsize=(width * grid_size[1], 1.6 * grid_size[0]), **figure_kwargs
    )
    axs = np.array(axs).flatten()
    assert len(axs) >= len(psams)
    for name, ax, psam in zip(names, axs, psams):
        if psam is None:
            assert name is None
            continue
        render_psam(psam, ax=ax, **kwargs)
        ax.set_title(name)
        if axes_mode == "completely_blank":
            ax.axis("off")
        elif axes_mode == "just_y":
            clean(ax)
        else:
            raise ValueError(axes_mode)
    if same_ylim:
        ylim = np.array([ax.get_ylim() for ax in axs]).flatten()
        ylim = [np.min(ylim), np.max(ylim)]
        for ax in axs:
            ax.set_ylim(*ylim)
    return axs


def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    # ax.spines["left"].set_visible(False)
    ax.set_xticks([])
    # ax.set_yticks([])
    ax.set_xlim(-1, ax.get_xlim()[1])
