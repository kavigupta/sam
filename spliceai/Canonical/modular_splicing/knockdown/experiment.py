"""
The main experiment, that goes in the paper
"""

from functools import lru_cache
from .analytics import Analysis, knockdown_analysis
from .models import short_donor_models
from .renderer import plot_knockdown_results

line_class = ("HepG2", "K562")

agg_spec = dict(
    type="StandardAggregator",
    query_combinator_spec=dict(type="multiply_with_powers"),
    difference_spec=dict(type="log_ratio"),
)

annotation_spec = dict(
    type="InSilicoDeltaPsiAnnotator",
    agg_spec=agg_spec,
)

analyses = [
    Analysis(True, "FDR < 0.05"),
    Analysis(True, "FDR < 0.25"),
    Analysis(False, "FDR < 0.25"),
    Analysis(True, "min_count > 50"),
    Analysis(False, "min_count > 50"),
    Analysis(True, "|delta_psi| > 0.25"),
    Analysis(False, "|delta_psi| > 0.25"),
    Analysis(True, "|delta_psi| > 0.1"),
    Analysis(False, "|delta_psi| > 0.1"),
]


@lru_cache(None)
def knockdown_results():
    """
    Gather the results for the given analyses. This is a slow operation, so we
    cache the results in memory in case you want to refine the plots.
    """
    models = short_donor_models()
    return knockdown_analysis(
        models, line_class, analyses=analyses, annotation_spec=annotation_spec
    )


def plot_knockdown_experiment(ax_dir, ax_mag, *, include_sai=False):
    """
    Plot the results of the knockdown experiment, on the given axes.

    Args:
        ax_dir: matplotlib axis, for the directional analyses
        ax_mag: matplotlib axis, for the magnitude analyses
        include_sai: bool, whether to include the SAI models
    """
    results = knockdown_results()
    color_for_model_group = {
        "FM": "blue",
        "AM": "red",
        "FM/sai": "purple",
        "AM/sai": "green",
    }
    model_groups = {
        model.split("_")[0]: [
            x for x in results if x.split("_")[0] == model.split("_")[0]
        ]
        for model in results
    }
    if not include_sai:
        model_groups = {k: v for k, v in model_groups.items() if not k.endswith("/sai")}
    for ax, analysis_to_plot in zip((ax_dir, ax_mag), ("FDR < 0.05", "min_count > 50")):
        if ax is None:
            continue
        plot_knockdown_results(
            ax,
            results,
            model_groups=model_groups,
            color_for_model_group=color_for_model_group,
            analysis_to_plot=[analysis_to_plot],
        )
