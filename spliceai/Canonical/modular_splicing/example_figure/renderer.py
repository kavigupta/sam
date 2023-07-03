from tempfile import NamedTemporaryFile

from permacache import permacache, stable_hash

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

from modular_splicing.motif_names import get_motif_names
from modular_splicing.utils.construct import construct
from modular_splicing.psams.motif_types import motifs_types

object_colors = {
    "3'": "#c77a0e",
    "5'": "#740ec7",
    "Motif": "#0e93c7",
}


def motif_names_for_species(species):
    if species == "human" or species == "mouse":
        return get_motif_names("rbns")
    if species == "fly":
        return sorted(
            construct(
                motifs_types(),
                dict(
                    type="rna_compete_motifs_for",
                    species_list=["Drosophila_melanogaster"],
                ),
            )
        )
    raise ValueError(f"Unknown species {species}")


def lolipop(ax, y, *, color, bottom, filled_circles=True, **kwargs):
    """
    Plot a lolipop plot, with the given y values.
    """
    top = max(y.max(), 0)
    if (y > bottom).any():
        _, lines, _ = ax.stem(
            y, **kwargs, linefmt=color, markerfmt="none", basefmt="none", bottom=bottom
        )
        lines.set_linewidth(0.5)
    x = np.arange(len(y))
    x, y = x[y > bottom], y[y > bottom]
    ax.scatter(
        x,
        y,
        s=10,
        **(
            dict(color=color)
            if filled_circles
            else dict(facecolors="none", edgecolors=color)
        ),
    )
    set_ylim(ax, bottom, top)


def set_ylim(ax, bottom, top):
    return ax.set_ylim(bottom, top + (top - bottom) * 0.2)


def plot_3_and_5prime(ax, y, yp, *, highlight_above, range_multiplier, **kwargs):
    """
    Plot 3' and 5' scores. Limit of -10 enforced to avoid cluttering.
    """
    bottom = -10 * range_multiplier
    assert len(yp.shape) == 2 and yp.shape[1] == 2
    for pos in np.where(y)[0]:
        ax.scatter(pos, bottom, color="black", s=500, marker="^")

    for i, name in enumerate(["3'", "5'"]):
        lolipop(ax, yp[:, i], color=object_colors[name], bottom=bottom, **kwargs)

    for pos, feat in zip(*np.where(yp > highlight_above)):
        ax.scatter(pos, yp[pos, feat], s=100, facecolors="none", edgecolors="k")
        # cir = plt.Circle((pos, y[pos, feat]), HIGHLIGHT_RADIUS, color="r", fill=False)
        # ax.add_patch(cir)
    ax.set_ylabel("3', 5' score [normalized]")
    ax.set_ylim(bottom, 1)
    ax.set_yticks(-np.arange(0, -bottom + 1, 2))
    ax.axhline(-1, color="black", linestyle=":", linewidth=0.5)


def plot_motifs(ax, mot, fill_circle, effects, *, species):
    """
    Plot the motif scores. Annotate the motifs that show up in `effects`.
    """
    names = motif_names_for_species(species)
    fill_circle_pos = fill_circle.any(-1)
    empty_circle_pos = ~fill_circle_pos
    lolipop(ax, mot.max(-1) * fill_circle_pos, color=object_colors["Motif"], bottom=0)
    lolipop(
        ax,
        mot.max(-1) * empty_circle_pos,
        color=object_colors["Motif"],
        bottom=0,
        filled_circles=False,
    )
    set_ylim(ax, 0, mot.max())
    for eff in effects:
        x, i = eff["mot_pos"], eff["mot_id"]
        ax.text(x, mot[x, i] + 2, names[i], rotation=90, fontsize=5)
    ax.set_ylabel("motif score [arb. units]")


def legend(ax):
    """
    Create a legend for the lolipop plots.
    """

    def circle(fill_color, is_filled=True, scale=1):
        kwargs = (
            dict(markerfacecolor=fill_color, markersize=5 * scale)
            if is_filled
            else dict(markeredgecolor=fill_color, markersize=3.5 * scale)
        )
        return mpl.lines.Line2D([], [], color="white", marker="o", **kwargs)

    legend_elements = []
    for k in ["3'", "5'"]:
        legend_elements.append((f"Potential {k} site", circle(object_colors[k])))

    legend_elements.append(
        (
            "Predicted site",
            circle("black", is_filled=False, scale=2),
        )
    )

    legend_elements.append(("Motif in both AM/FM", circle(object_colors["Motif"])))
    legend_elements.append(
        ("Motif in just one", circle(object_colors["Motif"], is_filled=False))
    )

    def make_legend_arrow(
        legend, orig_handle, xdescent, ydescent, width, height, fontsize
    ):
        return orig_handle

    for name, color in [("+ effect", "green"), ("- effect", "red")]:
        legend_elements.append(
            (
                name,
                create_arrow(
                    0,
                    0,
                    15 + 5 * (color == "green"),
                    0,
                    color=color,
                    repel_dist=0,
                    use_rad=False,
                ),
            )
        )

    legend_elements.append(
        (
            "Real splice site",
            mpl.lines.Line2D(
                [], [], color="white", marker="^", markerfacecolor="black", markersize=8
            ),
        )
    )

    names, els = zip(*legend_elements)
    ax.legend(
        els,
        names,
        loc="lower left",
        handler_map={
            mpl.patches.FancyArrowPatch: mpl.legend_handler.HandlerPatch(
                patch_func=make_legend_arrow
            )
        },
    )


def plot_effects(ax_m, ax_s, effects, mot, res, *, highlight_above):
    """
    Plot the effects from the motifs to the 3' and 5' scores.
    """
    for eff in effects:
        my = mot[eff["mot_pos"], eff["mot_id"]]
        plot_arrow(
            ax_m,
            ax_s,
            eff["mot_pos"],
            my,
            eff["feat_pos"],
            res[eff["feat_pos"], eff["feat_id"]],
            color="red" if eff["log_eff"] < 0 else "green",
            repel_dist=3 * (res[eff["feat_pos"], eff["feat_id"]] > highlight_above),
            dashed=not eff["above_thresh"],
        )


def plot_arrow(ax_start, ax_end, x1, y1, x2, y2, color, *, repel_dist, dashed=False):
    """
    Plot an arrow from (x1, y1) to (x2, y2) on the axes ax_start and ax_end.
    """
    y2 = rescale(*ax_end.get_ylim(), *ax_start.get_ylim(), y2)
    if not np.isfinite([y1, y2]).all():
        return
    arrow = create_arrow(x1, y1, x2, y2, color, repel_dist)
    ax_start.add_patch(arrow)
    if dashed:
        arrow = create_arrow(x1, y1, x2, y2, "white", repel_dist, trace_dashes=True)
        ax_start.add_patch(arrow)


def create_arrow(
    x1, y1, x2, y2, color, repel_dist, *, use_rad=True, trace_dashes=False
):
    if color == "green":
        style = "simple, head_width=3, head_length=6"
    else:
        style = "-[, widthB=3, lengthB=0.1"
    if trace_dashes:
        style = "-"
    arrow = mpl.patches.FancyArrowPatch(
        (x1, y1),
        (x2 - repel_dist * np.sign(x2 - x1), y2),
        connectionstyle=f"arc3,rad={-.3 * np.sign(x2 - x1) * use_rad}",
        arrowstyle=style,
        linewidth=0.5 * (1.3 if trace_dashes else 1),
        linestyle=(5, (10, 5)) if trace_dashes else "-",
        color=color,
    )

    return arrow


def rescale(start_low, start_high, end_low, end_high, y):
    """
    Rescale y from the start range to the end range.
    """
    y = (y - start_low) / (start_high - start_low)
    return y * (end_high - end_low) + end_low


def plot_splicepoints_motifs_effects(
    ax, ex, model_key, other_key, *, species, range_multiplier
):
    """
    Plot the splicepoints, motifs, and effects, for the given example.
    """
    ax_s, ax_m = ax, ax.twinx()
    plot_3_and_5prime(
        ax_s,
        ex["y"],
        ex[model_key]["res"],
        highlight_above=-1,
        range_multiplier=range_multiplier,
    )
    plot_motifs(
        ax_m,
        ex[model_key]["mot"],
        ex[other_key]["mot"] != 0,
        ex[model_key]["effects"],
        species=species,
    )
    plot_effects(
        ax_m,
        ax_s,
        ex[model_key]["effects"],
        ex[model_key]["mot"],
        ex[model_key]["res"],
        highlight_above=-1,
    )


def setup_xaxis(ax, ex, every=50):
    chrom, zero = ex["gene_pos"]["chrom"], ex["gene_pos"]["start"]
    gene = ex["gene_pos"]["gene_idx"]

    exstart = zero + ex["s"]
    exend = zero + ex["e"]

    fake = "true" if ex["y"][[ex["s"], ex["e"]]].any() else "false"

    label = f"Offset from {fake} exon {chrom}:{exstart}-{exend} in gene {gene}"
    ax.set_xlabel(label)

    xs = []
    xs_labels = []
    for x in reversed(range(every, ex["s"] + 1, every)):
        xs.append(ex["s"] - x)
        xs_labels.append(f"-{x}")
    xs.append(ex["s"])
    xs_labels.append("acc")
    for x in range(every, ex["e"] - ex["s"] - every // 2, every):
        xs.append(ex["s"] + x)
        xs_labels.append(f"{x}")
    xs.append(ex["e"])
    xs_labels.append("don")
    for x in range(every, len(ex["y"]) - ex["e"], every):
        xs.append(ex["e"] + x)
        xs_labels.append(f"+{x}")
    ax.set_xticks(xs, xs_labels)


def main_example_figure(ex, name, *, species, range_multiplier=1):
    """
    Plot the example figure for the given example.
    """
    im = render_main_example_figure(ex, species, range_multiplier)
    if name is not None:
        im.save(f"output-csvs/main-figure/{name}.png")
    return im


@permacache(
    "modular_splicing/example_figure/renderer/render_main_example_figure_3",
    key_function=dict(ex=stable_hash),
)
def render_main_example_figure(ex, species, range_multiplier):
    """
    Plot the example figure for the given example, and return the figure.
    """
    _, (ax_lssi, ax_fm, ax_am) = plt.subplots(
        3, 1, figsize=(10, 7.5), dpi=200, sharex=True
    )

    legend(ax_lssi)

    ax_lssi.set_title("LSSI")
    plot_3_and_5prime(
        ax_lssi,
        ex["y"],
        ex["spl"],
        highlight_above=-1,
        range_multiplier=range_multiplier,
    )

    ax_fm.set_title("FM")
    plot_splicepoints_motifs_effects(
        ax_fm, ex, "FM", "AM", species=species, range_multiplier=range_multiplier
    )

    ax_am.set_title("AM")
    plot_splicepoints_motifs_effects(
        ax_am, ex, "AM", "FM", species=species, range_multiplier=range_multiplier
    )

    setup_xaxis(ax_am, ex)

    with NamedTemporaryFile(suffix=".png") as f:
        plt.savefig(f.name, facecolor="white")
        plt.close()
        return Image.open(f.name)
