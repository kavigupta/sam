import numpy as np
import matplotlib.patches as mpatches

from .analytics import Analysis


def plot_knockdown_results(
    ax, results, *, model_groups, color_for_model_group, analysis_to_plot, alpha=0.5
):
    """
    Plot knockdown results.

    Args:
        ax: matplotlib axis
        results: dict[model name -> dict[Analysis -> confusion matr[2, 2]]]
        model_groups: dict[group name -> list[model names]]
        color_for_model_group: dict[group name -> color]
        analysis_to_plot: list[analysis filter names]
        alpha: float, alpha for the CI bars on the plots
    """

    model_names = [x for xs in model_groups.values() for x in xs]

    results = {k: results[k] for k in model_names}

    mlhs = []
    xs = []
    colors = []

    xs_centers = []
    for i, filter in enumerate(analysis_to_plot):
        xc = i * len(model_groups) * 2
        xs_centers.append(xc)

        xaround = np.arange(len(model_groups))
        xaround = xc + xaround - xaround.mean()
        for x, model_group_name in zip(xaround, model_groups):
            model_names_each = model_groups[model_group_name]
            xs_array = np.linspace(0, 0.2, len(model_names_each))
            xs_array -= xs_array.mean()
            xs.extend(xs_array + x)
            colors.extend(
                [color_for_model_group[model_group_name]] * len(model_names_each)
            )
            mlh = [
                results[k][
                    Analysis(
                        is_directional=filter.startswith("FDR"),
                        filter=filter,
                    )
                ]
                .accuracy_overall()
                .asarray()
                for k in model_names_each
            ]
            mlhs.extend(mlh)
    xs = np.array(xs)
    mean, low, hi = np.array(mlhs).T * 100
    ax.scatter(xs, mean, color=colors, alpha=alpha, marker=".")
    ax.errorbar(
        xs, (hi + low) / 2, (hi - low) / 2, ecolor=colors, alpha=alpha, linestyle=" "
    )
    ax.set_xticks(
        xs_centers,
        [
            {
                "FDR": "directional",
                "min_count": "magnitude",
            }[f.split()[0]]
            for f in analysis_to_plot
        ],
    )
    ax.set_ylabel("Accuracy [%]")
    ax.axhline(50, color="black")
    ax.grid()
    ax.legend(
        handles=[
            mpatches.Patch(color=color_for_model_group[mgn], label=mgn)
            for mgn in model_groups
        ]
    )
