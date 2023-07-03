import attr

import numpy as np
import matplotlib.pyplot as plt


def plot_base_importance(
    x,
    *importance,
    labels=[],
    chunk_size=81,
    features={},
    style={},
    headroom=0,
    yticks=None,
    yticksnames=None,
    dpi=400,
    feature_text_size=8,
    width=14,
):
    importance = np.array(importance)
    assert len(x.shape) == 2 and len(importance.shape) == 2

    nchunks = (x.shape[0] + chunk_size - 1) // chunk_size

    _, axs = plt.subplots(
        ncols=1, nrows=nchunks, figsize=(width * 2, nchunks * 2), dpi=dpi
    )
    axs = np.array([axs]).flatten()

    xpoints = np.arange(importance.shape[1]) + (x.shape[0] - importance.shape[1]) / 2
    bases = np.arange(x.shape[0])
    baselabels = render_with_importance(np.array([x]))[0]

    mini, maxi = np.nanmin(importance), np.nanmax(importance)
    mini -= headroom
    maxi += headroom
    if mini == maxi:
        mini -= 1
        maxi += 1

    if not isinstance(style, list):
        style = [style] * len(importance)

    for k in range(nchunks):
        chunk = slice(k * chunk_size, (k + 1) * chunk_size)
        for i, imp in enumerate(importance):
            label = None if i >= len(labels) else labels[i]
            axs[k].plot(xpoints[chunk], imp[chunk], label=label, **style[i])
        axs[k].set_xticks([bases[chunk][0] - 5, *bases[chunk]])
        axs[k].set_xticklabels([str(k * chunk_size) + ":", *baselabels[chunk]])
        axs[k].set_ylim(mini, maxi)
        axs[k].xaxis.set_ticks_position("none")
        if yticks is not None:
            axs[k].set_yticks(yticks)
            axs[k].set_yticklabels(yticksnames or yticks)
        axs[k].grid(axis="y")
        count = 0
        for fx in features:
            for index, feat in enumerate(features[fx]):
                if k * chunk_size <= fx < (k + 1) * chunk_size:
                    axs[k].axvline(fx, **feat.style)
                    count += 1
                    headroom = 0.15
                    yvalue = (1 - (count % 5) / 5) * (1 - headroom * 2) + headroom
                    axs[k].text(
                        fx,
                        mini + (maxi - mini) * yvalue,
                        feat.short_render(),
                        rotation=45,
                        fontsize=feature_text_size,
                    )
    if labels:
        plt.legend()


def render_with_importance(x):
    unimportant = "nacgt"
    indices = ((np.argmax(x, axis=2) + 1) * np.sum(x, axis=2)).astype(np.uint8)
    rendered = np.array(unimportant)[indices]
    return rendered


@attr.s
class Motif:
    name = attr.ib()
    color = attr.ib(default="grey")
    extra_style = attr.ib(default=attr.Factory(dict))

    def short_render(self):
        return self.name

    @property
    def style(self):
        return dict(color=self.color, **self.extra_style)
