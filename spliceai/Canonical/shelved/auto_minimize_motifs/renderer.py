from colorsys import hsv_to_rgb
import numpy as np
import matplotlib.pyplot as plt


def plot_text(x, ys, text, align, name):
    for i in range(ys.shape[0]):
        plt.text(x, ys[i], text[i], ha=align, va="center")

    plt.text(x, 1 + ys.shape[0], name, ha=align, va="center")


def color(name, names):
    names = sorted(names)
    rgb = hsv_to_rgb(names.index(name) / len(names) * 240 / 360, 1, 1)
    rgb = np.array(rgb)
    rgb /= rgb.sum()
    return rgb


def plot_relative_ordering(
    left,
    right,
    *,
    graph_start,
    graph_end,
    text_loc,
    line_loc,
    tick_gap,
    out_path=None,
    left_name,
    right_name,
    title=None,
    order=-1,
    show=True,
):
    mask = np.isfinite(left) & np.isfinite(right)
    left, right = left[mask], right[mask]

    plt.figure(figsize=(15, 15))
    full_data = np.array([left, right])
    lo, hi = np.nanmin(full_data, 1), np.nanmax(full_data, 1)

    transform = (
        lambda x, i: ((x - lo[i]) / (hi[i] - lo[i])) * (graph_end - graph_start)
        + graph_start
    )

    left_order, right_order = np.argsort(order * left), np.argsort(order * right)
    left_show, right_show = list(left[left_order].index), list(right[right_order].index)
    ys = np.arange(left.shape[0])

    plt.scatter(-transform(np.array(left)[left_order], 0), ys, color="black")
    plt.scatter(transform(np.array(right)[right_order], 1), ys, color="black")
    ticks = [np.arange(lo[i] // tick_gap * tick_gap, hi[i], tick_gap) for i in range(2)]
    numeric_ticks = [*-transform(ticks[0], 0), *transform(ticks[1], 1)]
    label_ticks = np.concatenate(ticks)
    plt.xticks(numeric_ticks, label_ticks)
    plt.yticks(ys - 0.5, [""] * len(ys))
    plot_text(-text_loc, ys, left_show, "right", left_name)
    plot_text(text_loc, ys, right_show, "left", right_name)

    for name in left_show:
        plt.plot(
            [-line_loc, line_loc],
            [left_show.index(name), right_show.index(name)],
            color=color(name, left_show),
            alpha=0.8,
            lw=3,
        )
    if title is not None:
        plt.title(title)
    plt.grid()
    if out_path is not None:
        plt.savefig(out_path, facecolor="white")
    if show:
        plt.show()
    else:
        plt.close()
