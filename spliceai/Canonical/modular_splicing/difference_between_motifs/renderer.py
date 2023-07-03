import numpy as np
import matplotlib.pyplot as plt

from matplotlib_venn import venn3

from modular_splicing.utils.plots.plot_psam import clean, render_psam
from .difference_classifier import bag_to_set, venn3_intersect_bag_sizes


def display_difference(
    results,
    matches_test,
    amount,
    *,
    motif_names,
    motifs_to_use,
    difference_mode,
    **kwargs,
):
    """
    Display the difference between motifs for the given PSAMs.
    """
    table = produce_difference_table(
        results,
        motif_names=motif_names,
        motifs_to_use=motifs_to_use,
        difference_mode=difference_mode,
    )
    display_difference_table(
        table, matches_test, motif_names, motifs_to_use, amount, **kwargs
    )


def produce_difference_table(results, *, motif_names, motifs_to_use, difference_mode):
    """
    Produce a table of the difference between motifs for the given PSAMs.
    """
    table = []
    for motif in motifs_to_use:
        motif_idx = motif_names.index(motif)
        column, column_names = [], []
        column += [results["AM_1"][motif_idx]["fm_psam"]]
        column_names += ["FM"]
        for k in results:
            column += [results[k][motif_idx]["am_psam"]]
            column_names += [k]
        column = [
            x * (2 + (x * np.log(x + 1e-100) / np.log(2)).sum(-1)[:, None])
            for x in column
        ]
        for k in results:
            p = compute_difference_psam(results[k][motif_idx], mode=difference_mode)
            column += [p]
            column_names += [
                f'{k} vs FM [{results[k][motif_idx]["eval_accuracy"]:.0%}]'
            ]

        column_names = [f"{motif}: {n}" for n in column_names]
        table.append([column, motif, column_names])
    return table


def compute_difference_psam(result, *, mode):
    if mode == "parameters_raw":
        p = result["discriminator_model"].coef_.reshape(21, 4)
        p = p - p.mean(-1)[:, None]
        return p
    elif mode == "parameters_scaled_to_entropy":
        p = compute_difference_psam(result, mode="parameters_raw")
        p = (
            p
            * (
                scaled_pointwise_mutual_information_difference(result).sum(-1)
                / np.abs(p).sum(-1)
            )[:, None]
        )
        return p
    elif mode == "individual_contributions":
        p = scaled_pointwise_mutual_information_difference(result)
        p *= np.sign(compute_difference_psam(result, mode="parameters_raw"))
        return p
    else:
        raise ValueError(f"Unknown mode: {mode}")


def scaled_pointwise_mutual_information_difference(result):
    def h(table):
        table += 1e-10
        return -table * np.log(table) / np.log(2)

    p_x_given_1, p_x_given_0 = [result[f"{k}m_only_psam"] for k in "af"]
    p_1 = p_0 = 0.5
    p_x = p_x_given_1 * p_1 + p_x_given_0 * p_0
    h_x_given_y = p_1 * h(p_x_given_1) + p_0 * h(p_x_given_0)
    h_x = h(p_x)
    return h_x - h_x_given_y


def draw_venn_diagram(matches_test, *, amount, motif_idx, ax):
    venn_labels = "FM_1", "AM_1", "AM_2"
    assert set(matches_test.keys()) == set(venn_labels)
    counts_each = [matches_test[k][motif_idx].size / amount for k in venn_labels]
    sets = venn3_intersect_bag_sizes(
        *[bag_to_set(matches_test[k][motif_idx]) for k in venn_labels]
    )
    sets = {k: v / amount for k, v in sets.items()}
    venn3(
        sets,
        subset_label_formatter="{:.3%}".format,
        set_labels=[
            x.replace("FM_1", "FM") + f" [{c:.2%}]"
            for x, c in zip(venn_labels, counts_each)
        ],
        ax=ax,
    )


def display_difference_table(
    table,
    matches_test,
    motif_names,
    motifs_to_use,
    amount,
    *,
    size_each=8,
    padding=1,
    line_margin=0,
    line_relative_placement=0.5,
):
    """
    Display the difference between motifs for the given PSAMs.
    """
    assert size_each % 2 == 0
    stride = size_each + padding

    fig = plt.figure(figsize=(20, 20), dpi=100, tight_layout=True, facecolor="white")

    column_spans = [5] * 4
    column_locations = np.cumsum([1] + column_spans)[:-1]

    gridsize = len(table) * stride - padding, 1 + sum(column_spans)

    for j in range(4):
        ax = plt.subplot2grid(
            gridsize, (0, column_locations[j]), colspan=column_spans[j]
        )
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            ["FM (acc=67%)", "AM (acc=79%)", "Difference", "Bound Locations"][j],
            ha="center",
            va="center",
            size=30,
        )

    for i in range(len(table)):
        ax = plt.subplot2grid(
            gridsize, (1 + i * stride, 0), rowspan=size_each, colspan=1
        )
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            f'({"abc"[i]})',
            ha="center",
            va="center",
            size=20,
            # rotation=90,
            rotation_mode="anchor",
        )

    for i, (psams, overall_name, ns) in enumerate(table):
        ax_fm = plt.subplot2grid(
            gridsize,
            (1 + i * stride + size_each // 4, column_locations[0]),
            rowspan=size_each // 2,
            colspan=column_spans[j],
        )
        render_psam(psams[0], ax=ax_fm, psam_mode="raw")
        ax_fm.set_ylabel("Information [b]")
        ax_fm.set_xlabel("Position in motif")
        clean(ax_fm)
        ax_fm.set_title(ns[0])
        #     ax_fm.set_ylabel(overall_name)
        psams, ns = psams[1:], ns[1:]
        for j in range(2):
            for k in range(2):
                ax_am = plt.subplot2grid(
                    gridsize,
                    (1 + i * stride + k * size_each // 2, column_locations[j + 1]),
                    rowspan=size_each // 2,
                    colspan=column_spans[j + 1],
                )
                render_psam(psams[j * 2 + k], ax=ax_am, psam_mode="raw")
                ax_am.set_title(ns[j * 2 + k])
                mi_cal, ma_cal = minmax(psams[1 * 2 + k])
                _, ma_cur = minmax(psams[j * 2 + k])
                ax_am.set_ylim(mi_cal / ma_cal * ma_cur, ma_cur)
                ax_am.set_ylabel("Information [b]")
                ax_am.set_xlabel("Position in motif")
                clean(ax_am)
        if i != 0:
            level = (
                -1 + i * stride - padding + line_relative_placement * padding
            ) / gridsize[0]
            line = plt.Line2D(
                (line_margin, 1 - line_margin), (level, level), color="k", linewidth=0.5
            )
            fig.add_artist(line)

    for i in range(len(table)):
        ax = plt.subplot2grid(
            gridsize,
            (1 + i * stride, column_locations[-1]),
            rowspan=size_each,
            colspan=column_spans[-1],
        )
        draw_venn_diagram(
            matches_test,
            amount=amount,
            motif_idx=motif_names.index(motifs_to_use[i]),
            ax=ax,
        )
    plt.tight_layout()


def highest_lower_power_of_2(x):
    return 2 ** np.floor(np.log2(x))


def minmax(p):
    return (p * (p < 0)).sum(-1).min(), (p * (p > 0)).sum(-1).max()


def cross_classify(results):
    """
    Display the cross-classification accuracy of the discriminators.
    """
    for k, result in results.items():
        xs, ys = zip(
            *[
                (
                    o["eval_accuracy"] * 100,
                    np.mean(
                        [
                            o["eval_accuracies"][kprime]
                            for kprime in results
                            if kprime != k
                        ]
                    )
                    * 100,
                )
                for o in result
                if o["discriminator_model"] is not None
            ]
        )
        plt.scatter(
            list(xs) + [np.mean(xs)],
            list(ys) + [np.mean(ys)],
            alpha=0.5,
            s=[20] * len(xs) + [200],
            label=f"{k} [mean=({np.mean(xs)/100:.0%}, {np.mean(ys)/100:.0%})]",
        )
    plt.xlabel("Eval acc on same motif [%]")
    plt.ylabel("Eval acc on other motif [%]")
    plt.plot([75, 100], [75, 100], color="black")
    plt.grid()
    plt.legend()
    plt.show()
