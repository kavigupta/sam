import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

from modular_splicing.motif_names import get_motif_names
from modular_splicing.motif_perturbations.perturbations_on_standardized_sample import (
    all_mpi_on_standardized_sample,
)
from modular_splicing.motif_perturbations.summarize_effect import (
    positional_effects_near_splicepoints,
)
from .well_understood_proteins import remap_motif, should_use_motif
from .van_nostrand_data import line_up_vn_with_ours


def get_raw_effects(mod, effect_radius, amount=64_000, blur_radius=0):
    """
    Get the effects of our model on splicing. Is the mean over all AM models.

    Arguments:
        effect_radius: the radius of the region around the splicing site to
            consider.

    Returns
    -------
    effects: (4, 2 * effect_radius + 1, M)
        Formatted as in the output of positional_effects_near_splicepoints.
    """
    names = get_motif_names("rbns")
    models = mod.binarized_models()
    perturbations = all_mpi_on_standardized_sample(
        models, always_use_fm=False, is_binary=True, sl=1000, amount=amount
    )

    effects_ours = {
        m.name: positional_effects_near_splicepoints(
            perturbations[m.name],
            num_motifs=len(names),
            blur_radius=blur_radius,
            effect_radius=effect_radius,
            normalize_mode="by_motifs_presence",
        )
        for m in models
    }
    effects_ours = np.array([effects_ours[k] for k in effects_ours]).mean(0)
    return effects_ours


def select_motifs(effect, ensure_contains_motifs):
    """
    Select effects of the motifs we think make sense to include in the plot.
    """
    # create a nan row to put in for the padding motifs
    nan_row = effect[:, :, 0] + np.nan
    effect = {
        remap_motif(name): effect[:, :, i]
        for i, name in enumerate(get_motif_names("rbns"))
        if should_use_motif(name)
    }
    for name in ensure_contains_motifs:
        if name not in effect:
            effect[name] = nan_row
    return (
        np.concatenate([effect[name][:, :, None] for name in sorted(effect)], axis=2),
        sorted(effect),
    )


def convert_to_image(effect_ours, names, *, padding):
    """
    Convert the effects to the format of an image, by extracting the overall
        acceptor and donor effects, then concatenating these side by side
        (with a gap in between).

    Arguments:
        effect_ours: (4, L, M)
            Formatted as in the output of positional_effects_near_splicepoints.
        names: (M,)
            the names of the motifs.
        padding:
            How much padding to add between the acceptor and donor effects.
    Returns
    -------
        effect: (M, L + padding + L)
            The effect of the motifs on splicing, ready to plot as an impage
            with imshow.
        xaxis:
            kwargs for the xaxis ticks and labels.
        summary:
            Summary of the intron and exon effects, in case we want to present
            these as a table.
    """
    pos_acc, neg_acc, pos_don, neg_don = effect_ours
    eff_acc, eff_don = pos_acc - neg_acc, pos_don - neg_don
    radius = eff_acc.shape[0] // 2
    whole = eff_acc.shape[0]
    assert radius * 2 + 1 == whole

    eff_intron = eff_acc[:radius].sum(0) + eff_don[-radius:].sum(0)
    eff_exon = eff_don[:radius].sum(0) + eff_acc[-radius:].sum(0)

    # concatenated acc results then don results
    eff_overall = np.concatenate(
        [eff_acc, np.zeros((padding, eff_acc.shape[1])) + np.nan, eff_don]
    ).T

    xaxis = []
    xnames = []
    for anchor, anchor_name in [(radius, "A"), (whole + padding + radius, "D")]:
        for off in [-100, -50, 0, 50, 100]:
            xaxis.append(anchor + off)
            xnames.append(
                anchor_name + ("" if off == 0 else ("+" * (off > 0)) + str(off))
            )

    return (
        eff_overall,
        dict(ticks=xaxis, labels=xnames),
        pd.DataFrame(dict(intron=eff_intron, exon=eff_exon), index=names),
    )


def add_vertical_padding(effect, names, data_height, gap_height):
    """
    Inserts a gap between the motifs, to make the plot more readable.

    Arguments:
        effect: (M, L')
            the image representing the effects of the motifs.
        names: (M,)
            the names of the motifs.
        data_height: int
            the height of each data line, in pixels. Absolute value is irrelevant,
            only the ratio between data_height and gap_height matters.
        gap_height: int
            the height of the gap between the motifs, in pixels.

    Returns
    -------
        effect: (M * data_height + (M - 1) * gap_height, L')
            the image representing the effects of the motifs, with gaps.
        yaxis: dict
            kwargs for the yaxis ticks and labels.
    """
    effect_padded = []
    yticks = []
    for i in range(len(effect)):
        effect_padded += [effect[i]] * data_height
        if i != len(effect) - 1:
            effect_padded += [effect[i] + np.nan] * gap_height
        yticks += [i * (data_height + gap_height) + (data_height - 1) / 2]
    effect_padded = np.array(effect_padded)
    return effect_padded, dict(ticks=yticks, labels=names)


def rescale_effects(effect, global_scaling=False):
    """
    Rescale the effects per-motif such that 99% of them are between -1 and 1.

    Arguments:
        effect: (M, L')
            the image representing the effects of the motifs.
    Returns
    -------
        effect: (M, L')
            the image representing the effects of the motifs, rescaled.
    """
    if global_scaling:
        percentiles = np.nanpercentile(np.abs(effect), 99)
    else:
        percentiles = np.nanpercentile(np.abs(effect), 99, axis=1)[:, None]
    return effect / percentiles


def fully_padded_image(effect, ensure_contains_motifs):
    """
    Run all the steps in processing the effect into an image.

    Arguments:
        effect: the output of get_raw_effects.
        ensure_contains_motifs: list of str, motifs that must be present in the
            output, even if they aren't in RBNS

    Returns
    -------
        effect: (M * data_height + (M - 1) * gap_height, L')
            the image representing the effects of the motifs, with gaps, and rescaled.
        xaxis: dict
            kwargs for the xaxis ticks and labels.
        yaxis: dict
            kwargs for the yaxis ticks and labels.
        summary: dict
            Summary of the intron and exon effects, in case we want to present
            these as a table.
    """
    effect, names = select_motifs(effect, ensure_contains_motifs)
    effect, xaxis, summary = convert_to_image(effect, names, padding=25)
    effect, yaxis = add_vertical_padding(effect, names, data_height=5, gap_height=1)
    effect = rescale_effects(effect)

    return effect, xaxis, yaxis, summary


def plot_image(fig, ax, effect, xaxis, yaxis, do_cbar=True):
    """
    Plot the image of the effects.
    """
    im = ax.imshow(effect, aspect="auto", cmap="coolwarm")
    im.set_clim(-1, 1)
    ax.set_yticks(**yaxis)
    ax.set_xticks(**xaxis, rotation=90)
    ax.grid(axis="x", color="black")
    if do_cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax, orientation="vertical")
        cbar.ax.get_yaxis().labelpad = 15
        cbar.set_label("Effect [normalized]", rotation=270)


def experiment(model, mode="normal", effect_radius=100):
    """
    Run the experiment on a model.

    Arguments:
        model: a model series
        mode: "normal" or "van_nostrand_overlaps"
            normal: just plot the effect of the motifs under our model
            van_nostrand_overlaps: plot the effect of the motifs under our model,
                and also plot the vn rna maps, on motifs where both of us
                have the data.
            van_nostrand_all: plot the effect of the motifs under our model,
                and also plot the vn rna maps, on all motifs either
                of us have the data.
    """

    raw_effect = get_raw_effects(model, effect_radius=effect_radius)
    if mode == "normal":
        plt.figure(figsize=(10, 7), tight_layout=True, facecolor="white")
        return plot_effect(plt.gcf(), plt.gca(), raw_effect, ensure_contains_motifs=[])
    elif mode == "van_nostrand_overlaps":
        plot_both(raw_effect=raw_effect, only_overlaps=True)
    elif mode == "van_nostrand_all":
        plot_both(raw_effect=raw_effect, only_overlaps=False)
    else:
        raise ValueError(f"Unknown mode {mode}")


def plot_both(raw_effect, *, only_overlaps):
    plt.figure(figsize=(12, 7), tight_layout=True, facecolor="white")
    left_half, right_half = 5, 3

    ax = plt.subplot2grid(
        (1, left_half + right_half), (0, left_half), colspan=right_half
    )
    vn_names = van_nostrand_results(plt.gcf(), ax, only_ours=only_overlaps)
    ax.set_title("RNA Maps [Van Nostrand 2020]")

    ax = plt.subplot2grid((1, left_half + right_half), (0, 0), colspan=left_half)
    plot_effect(
        plt.gcf(),
        ax,
        raw_effect,
        ensure_contains_motifs=[] if only_overlaps else vn_names,
        do_cbar=False,
    )
    ax.set_title("Ours")


def plot_effect(fig, ax, raw_effect, ensure_contains_motifs, **kwargs):
    """
    Plot the effect of the motifs under our model, on the given axes.
    """
    effect, xaxis, yaxis, summary = fully_padded_image(
        raw_effect, ensure_contains_motifs
    )
    plot_image(fig, ax, effect, xaxis, yaxis, **kwargs)
    delta = summary.exon - summary.intron
    inconsistent = (delta < 0) != ([x.startswith("HN") for x in delta.index])
    return delta[inconsistent]


def van_nostrand_results(fig, ax, *, only_ours):
    """
    Create a plot of the results from Van Nostrand 2020, on the given axes.
    """
    our_names = {remap_motif(x) for x in get_motif_names("rbns") if should_use_motif(x)}
    image, names, xaxis = line_up_vn_with_ours(
        padding=20,
        ensure_contains=our_names,
        only_ours=only_ours,
    )
    effect, yaxis = add_vertical_padding(image, names, data_height=5, gap_height=1)
    effect = rescale_effects(effect)
    plot_image(fig, ax, effect, xaxis, yaxis)
    return names
