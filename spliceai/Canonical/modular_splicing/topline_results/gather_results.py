from modular_splicing.evaluation import standard_e2e_eval

from modular_splicing.lssi.maxent_tables import three_prime, five_prime
from modular_splicing.lssi.analyze import maxent_accuracies, lssi_accuracies
from modular_splicing.models_for_testing.list import (
    AM,
    spliceai_10k,
    LSSI,
    LSSI_EXTRAS,
)
from modular_splicing.models_for_testing.main_models import FM_full


def gather_results(replicated_models):
    """
    Produce accuracy results for the given dictionary of replicated models.

    Arguments:
        replicated_models: A dictionary mapping model names to lists of models.

    Returns:
        A dictionary mapping model names to lists of accuracies.
    """
    return {
        category: [
            standard_e2e_eval.evaluate_model_with_step(m)
            for m in replicated_models[category]
        ]
        for category in replicated_models
    }


def results_for_main_figure():
    """
    Get the accuracy results for the main figure.
    """
    all_accuracies = {}
    all_accuracies["MaxEnt"] = [maxent_accuracies([three_prime, five_prime])]
    all_accuracies["LSSI"] = [
        lssi_accuracies([x.model for x in le]) for le in [LSSI, *LSSI_EXTRAS]
    ]
    all_accuracies.update(
        gather_results(
            {
                "SpliceAI-10k": spliceai_10k,
                FM_full.name: FM_full.non_binarized_models(),
                AM.name: AM.non_binarized_models(),
            }
        )
    )
    return all_accuracies
