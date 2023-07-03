import matplotlib.pyplot as plt
import numpy as np

from .analysis import cl_for_model

from .evaluation import evaluate_models


def plot_results_by_channel(axs, results, channel_names, title):
    for c, name in enumerate(channel_names):
        for mname in results:
            if c < results[mname].shape[0]:
                axs[c].plot(results[mname][c] * 100, label=mname)
            else:
                axs[c].plot([], [], label=mname)

        axs[c].grid()
        axs[c].set_ylabel("Accuracy")
        axs[c].set_xlabel("Allowable offset")
        axs[c].set_title(f"{title}: {name}")
        axs[c].set_xscale("log")
    axs[-1].legend()


def render_branch_site_importance(f, idx):
    f = f.reshape(3, 2, *f.shape[1:])

    graphs = ["A", "D", "B"]
    which = [0, 1, 0]
    colors = ["red", "green", "blue"]

    xvals = np.arange(-(f.shape[2] // 2), 1 + f.shape[2] // 2)
    num_graphs = max(which) + 1

    size = 5
    _, axs = plt.subplots(1, num_graphs, figsize=(num_graphs * size, size))
    for c in range(f.shape[0]):
        ax = axs[which[c]]
        ax.plot(xvals, f[c, 0, :, idx] * 100, color=colors[c], label=graphs[c])
        ax.plot(xvals, -f[c, 1, :, idx] * 100, color=colors[c])
    for ax in axs:
        ax.grid()
        ax.legend()
        ax.set_xlabel("Offset from feature [nt]")
        ax.set_ylabel("Effect on feature [% of global effect]")
    plt.tight_layout()


def plot_all(model_specs, load_fn, title, *, delta_max=100):
    _, axs = plt.subplots(2, 3, figsize=(15, 10), tight_layout=True)
    for i, require_single_branch in enumerate((False, True)):
        plot_results_by_channel(
            axs[i],
            evaluate_models(
                model_specs,
                delta_max=delta_max,
                load_fn=load_fn,
                require_single_branch_site=require_single_branch,
                cl_for_model=cl_for_model,
            ),
            "ADB",
            title + " (only on 1 branch site)" * require_single_branch,
        )
    plt.tight_layout()
    plt.show()
