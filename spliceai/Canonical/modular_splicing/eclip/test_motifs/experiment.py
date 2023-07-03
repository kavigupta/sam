from functools import lru_cache

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from modular_splicing.eclip.test_motifs import sparsity_matched_coverage_results
from modular_splicing.eclip.trained_on_eclip.train import models_trained_on_eclip

from modular_splicing.fit_rbns.load_rbns_neural import load_rbns_models_for_evaluation

from modular_splicing.models_for_testing.list import AM, AM_sai
from modular_splicing.utils.plots.results_by_model_group import plot_by_model_group


@lru_cache(None)
def all_eclip_results():
    """
    All the results from the eclip experiments, for analysis. This is a list of dictionaries, one for each
    experiment.

    Cached (in memory) for speed.
    """
    rbns_79_sd = sparsity_matched_coverage_results(
        motif_names_source="rbns",
        test_models={m.name: m.model for m in AM.binarized_models()},
        data_amount=10_000,
    )

    rbns_79_sd_long_cl = sparsity_matched_coverage_results(
        motif_names_source="rbns",
        test_models={m.name: m.model for m in AM_sai.binarized_models()},
        data_amount=10_000,
        cl=400,
    )

    trained_on_rbns = sparsity_matched_coverage_results(
        motif_names_source="rbns",
        test_models=load_rbns_models_for_evaluation(just_nfm=True),
        data_amount=10_000,
        cl=400,
    )

    trained_on_eclip = sparsity_matched_coverage_results(
        motif_names_source="eclip_18",
        test_models=models_trained_on_eclip(motif_names_source="eclip_18"),
        data_amount=10_000,
    )

    trained_on_eclip = trained_on_eclip.copy()
    trained_on_eclip["out"] = {
        k.replace("trained_on_eclip_am_21x2_178_post_sparse_scale", "AM-E"): v
        for k, v in trained_on_eclip["out"].items()
    }

    return [rbns_79_sd, rbns_79_sd_long_cl, trained_on_rbns, trained_on_eclip]


def mean_enrichment():
    """
    Mean enrichment for each experiment.
    """
    mean_enrich = {}
    for table in all_eclip_results():
        mean_enrich.update(
            {
                k: dict(
                    iec=table["out"][k].intron_exon_controlled_mean_enrichment,
                    **table["out"][k].relative_mean_enrichment,
                )
                for k in table["out"]
            }
        )
    return pd.DataFrame(mean_enrich)


def plot_relative_enrichment(
    col,
    title,
    *,
    remove_spliceai_as_downstream,
    remove_nfm=False,
    ax=None,
    y_label="Difference in enrichment vs FMs [%]",
):
    """
    Plot relative enrichment

    Arguments:
        col: Which column to plot. Either "iec", "introns", or "exons"
        title: Title of the plot
        remove_spliceai_as_downstream: If True, remove spliceai as downstream experiments
        ax: Axis to plot on. If None, create a new figure.
    """
    mean_enrich = mean_enrichment()
    if remove_spliceai_as_downstream:
        mean_enrich = mean_enrich[
            [x for x in mean_enrich if "/sad" not in x and "/sai" not in x]
        ]
    if remove_nfm:
        mean_enrich = mean_enrich[[x for x in mean_enrich if "NFM" not in x]]
    if ax is None:
        plt.figure(dpi=200, figsize=(4, 4))
        ax = plt.gca()
    plot_by_model_group(
        np.array(mean_enrich.loc[col]) * 100,
        list(mean_enrich),
        title,
        y_label,
        colors={
            "NFM": "purple",
            "AM": "red",
            "AM/sai": "green",
            "AM-E": "gray",
        },
        ax=ax,
    )

    return {
        pre: mean_enrich[[x for x in mean_enrich if x.startswith(pre)]].loc[col].mean()
        for pre in ["AM_", "AM/", "AM [tr", "NFM"]
    }


def eclip_plot_for_paper():
    """
    Get the plot for the paper
    """
    _, axs = plt.subplots(1, 2, figsize=(4, 4), tight_layout=True, sharey=True, dpi=200)
    plot_relative_enrichment(
        "introns",
        "Average in Introns",
        remove_spliceai_as_downstream=True,
        ax=axs[1],
    )
    plot_relative_enrichment(
        "exons",
        "Average in Exons",
        remove_spliceai_as_downstream=True,
        ax=axs[0],
    )
    axs[1].set_ylabel("")


def small_eclip_plot_for_paper():
    """
    Get the plot for the paper
    """
    _, axs = plt.subplots(1, 2, figsize=(5, 3), tight_layout=True, sharey=True, dpi=200)
    plot_relative_enrichment(
        "introns",
        "Average in Introns",
        remove_spliceai_as_downstream=True,
        remove_nfm=True,
        ax=axs[1],
    )
    plot_relative_enrichment(
        "exons",
        "Average in Exons",
        remove_spliceai_as_downstream=True,
        remove_nfm=True,
        ax=axs[0],
    )
    axs[1].set_ylabel("")


def per_motif_table():
    """
    Get a table by motif
    """
    overall_by_motif = {}
    for t in all_eclip_results():
        overall_by_motif.update(
            {
                (k, k2): dict(zip(t["common_names"], v2))
                for k, v in t["out"].items()
                for k2, v2 in v.relative_enrichment_by_motif.items()
            }
        )
    return pd.DataFrame(overall_by_motif).T


def summary_for_paper():
    """
    Produces the summary information for the text of the paper
    """
    table = mean_enrichment()
    for row in table.index:
        print(row)
        enrich = table.loc[row]
        am = enrich[[x for x in enrich.index if x.startswith("AM_")]].mean()
        am_e = enrich[[x for x in enrich.index if x.startswith("AM-E")]].mean()
        nfm = enrich[[x for x in enrich.index if x.startswith("NFM")]].mean()
        print(f"The AMs have {am:.2%} relative enrichment over FM")
        print(f"The NFM has {nfm:.2%} relative enrichment over FM")
        print(f"The AM-Es have {am_e:.2%} relative enrichment over FM")
        am, nfm = am / am_e, nfm / am_e
        print(f"The AMs produce {am:.2%} of the enrichment of the AM-Es")
        print(f"The NFM produces {nfm:.2%} of the enrichment of the AM-Es")
        print()
