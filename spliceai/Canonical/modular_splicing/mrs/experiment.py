import numpy as np
import matplotlib.pyplot as plt
from modular_splicing.eclip.trained_on_eclip.train import models_trained_on_eclip
from modular_splicing.fit_rbns.load_rbns_neural import load_rbns_models_for_evaluation
from modular_splicing.mrs.analysis import paper_figure_analysis

from modular_splicing.utils.plots.plot_psam import render_psam

from modular_splicing.models_for_testing.list import FM, AM, FM_eclip_18
from modular_splicing.motif_names import get_motif_names

from .motifs import subset_delta_results
from .load import mrs_data


def models_and_seeds_for_main_analysis():
    """
    Get the models and seeds used in the main analysis.
    """
    seeds = (1, 2, 3, 4, 5)
    models = {
        "FM": FM.non_binarized_model(1).model,
        **{f"AM #{i}": AM.non_binarized_model(i).model for i in seeds},
    }
    return models, seeds


def psam_examples_tra2a():
    """
    Get an analysis of the TRA2A motif, as an example.
    """
    data = mrs_data()
    tra2a_index = get_motif_names("rbns").index("TRA2A")
    models, seeds = models_and_seeds_for_main_analysis()
    for_tra2a = {
        k: subset_delta_results(
            data[k]["y"], models, data[k]["x"], seeds, motif_idx=tra2a_index
        )[0]
        for k in data
    }
    render_psam(np.eye(4)[data[5]["x"]].mean(0))
    plt.axvspan(-0.5, -0.5 + 10, alpha=0.5, color="grey")
    plt.axvspan(-0.5 + 32, -0.5 + 32 + 10, alpha=0.5, color="grey")
    plt.axvspan(-0.5 + 10, -0.5 + 32, alpha=0.5, color="green")
    plt.plot(np.arange(10, 10 + 22), for_tra2a[5], color="black")
    plt.show()
    render_psam(np.eye(4)[data[3]["x"]].mean(0))
    plt.axvspan(-0.5, -0.5 + 10, alpha=0.5, color="grey")
    plt.axvspan(-0.5 + 35, -0.5 + 35 + 10, alpha=0.5, color="grey")
    plt.axvspan(-0.5 + 10, -0.5 + 35, alpha=0.5, color="green")
    plt.plot(np.arange(10, 10 + 25), for_tra2a[3], color="black")
    plt.show()


def mrs_statistics():
    """
    Print out the statistics for the MRS data
    """
    data = mrs_data()
    print(
        f'the 3\' dataset contains {data[3]["y"].shape[0] /1e6:.2f}m datapoints'
        + f' whereas the 5\' dataset contains only {data[5]["y"].shape[0] / 1e3:.0f}k datapoints'
    )
    print(
        f'the 3\' dataset is much less balanced, with RII having a standard deviation of {data[3]["y"].std():.2f}'
        + f' whereas the 5\' dataset has a standard deviation of {data[5]["y"].std():.2f}'
    )

    def relhist(y, title):
        plt.hist(
            y,
            bins=np.linspace(-1, 1, 21),
            alpha=0.5,
            label=title,
            weights=1 / y.shape[0] / np.ones_like(y),
        )

    relhist(data[5]["y"], "5'")
    relhist(data[3]["y"], "3'")
    plt.xlabel("Relative intronic enrichment")
    plt.ylabel("Relative Frequency")
    plt.legend()
    plt.show()


def main_analysis(axs=None):
    """
    Main analysis of the MRS data. Plot for the paper.
    """
    models, seeds = models_and_seeds_for_main_analysis()
    paper_figure_analysis(get_motif_names("rbns"), mrs_data(), models, seeds, axs=axs)


def nfm_analysis():
    """
    Control analysis of the MRS data. Plot not included in the paper.
    """
    models_on_rbns = {
        "FM": FM.non_binarized_model(1).model,
        **{
            k.replace("NFM_", "AM #"): v
            for k, v in load_rbns_models_for_evaluation(just_nfm=True).items()
        },
    }
    paper_figure_analysis(
        get_motif_names("rbns"),
        mrs_data(),
        models_on_rbns,
        (1, 2, 3, 4),
    )


def ame_analysis():
    """
    Control analysis of the MRS data (using AM-E). Plot not included in the paper.
    """
    ams = models_trained_on_eclip(motif_names_source="eclip_18")
    seeds = {int(x.split("_")[-1]) for x in ams}
    assert len(seeds) == len(ams)
    ams = {f"AM #{k.split('_')[-1]}": m for k, m in ams.items()}
    models_on_eclip = {
        "FM": FM_eclip_18.non_binarized_model(1).model,
        **ams,
    }
    paper_figure_analysis(
        get_motif_names("eclip_18"),
        mrs_data(),
        models_on_eclip,
        tuple(sorted(seeds)),
    )
