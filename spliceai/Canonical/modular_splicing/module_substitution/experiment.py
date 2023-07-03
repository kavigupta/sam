import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .evaluate_combinations import table_of_combinations

from modular_splicing.models_for_testing.list import AM
from modular_splicing.evaluation import standard_e2e_eval


def module_substitution_table():
    """
    Main table for the module substitution experiment.
    """
    result_df = table_of_combinations()
    display_table = {}
    for downstream_model in ["D=FM_1", "D=self"]:
        display_d = downstream_model.replace("FM_1", "FM")
        display_table[display_d] = {}
        for motif_model in [*[f"M=AM_{i}" for i in range(1, 1 + 5)], "M=FM_1"]:
            display_m = motif_model.replace("FM_1", "FM")
            display_table[display_d][display_m] = result_df.loc[
                motif_model,
                motif_model.replace("M=", "D=")
                if downstream_model == "D=self"
                else downstream_model,
            ]
    display_table = pd.DataFrame(display_table)
    return display_table


def module_substitution_main(ax=None):
    """
    Main plot for the module substitution experiment.
    """
    display_table = module_substitution_table()
    just_ams = display_table.loc[
        [x for x in display_table.index if x.startswith("M=AM")]
    ]
    just_fms = display_table.loc[
        [x for x in display_table.index if x.startswith("M=FM")]
    ]

    if ax is None:
        plt.figure(figsize=(4, 3), dpi=200)
        ax = plt.gca()
    labels = []
    kwargs = dict(alpha=0.5, s=100)
    ax.scatter(
        [len(labels)] * len(just_fms),
        just_fms["D=FM"],
        color="blue",
        **kwargs,
    )
    labels.append("FM")
    ax.scatter(
        [len(labels)] * len(just_ams),
        just_ams["D=self"],
        color="red",
        **kwargs,
    )
    labels.append("AM")
    ax.scatter(
        [len(labels)] * len(just_ams),
        just_ams["D=FM"],
        color="purple",
        **kwargs,
    )
    labels.append("AM w/FM Aggregator")
    ax.set_xticks(range(len(labels)), labels, rotation=0)
    ax.set_ylabel("Accuracy [%]")
    ax.set_ylim(65, 80)
    return just_fms.mean(), just_ams.mean()


def binarization_gap_statistics():
    """
    Statistics for the gap between binarized and non-binarized motifs.
    """
    display_table = module_substitution_table()
    original_self = [
        100 * np.mean(standard_e2e_eval.evaluate_model_with_step(m))
        for m in AM.non_binarized_models()
    ]
    binarized_self = np.array(
        display_table["D=self"][
            [x for x in display_table.index if x.startswith("M=AM")]
        ]
    )
    delta = original_self - binarized_self
    print(f"Min delta: {delta.min():.2f}%, max delta: {delta.max():.2f}%")
