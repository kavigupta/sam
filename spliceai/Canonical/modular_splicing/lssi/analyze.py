from permacache import permacache, stable_hash
import tqdm.auto as tqdm
import numpy as np
from modular_splicing.lssi.utils import second_half_of_spliceai_cached

from modular_splicing.utils.run_batched import run_batched

from modular_splicing.models.modules.lssi_in_model import LSSI_MODEL_THRESH
from modular_splicing.models_for_testing.list import LSSI
from .maxent import run_maxent_on_data
from .maxent_tables import three_prime, five_prime


@permacache(
    "modular_splicing/lssi/analyze/run_lssi_on_test_set_2",
    key_function=dict(model=stable_hash),
)
def run_lssi_on_test_set(model, channel):
    """
    Run the LSSI model on the test set and return the predictions and the labels.
    """
    xs, ys = second_half_of_spliceai_cached()
    yps = run_batched(
        lambda x: model(x).log_softmax(-1)[:, :, channel], xs, 128, pbar=tqdm.tqdm
    )
    cl_extra = yps.shape[1] - ys.shape[1]
    assert cl_extra > 0 and cl_extra % 2 == 0, str(cl_extra)
    trim = cl_extra // 2
    return ys, yps[:, trim:-trim]


def run_both_lssi_on_test_set(models):
    """
    Run the LSSI models on the test set and return the predictions and the labels

    Returns
    -------
    ys : np.ndarray
        The labels
    acc : np.ndarray
        The acceptor predictions
    don : np.ndarray
        The donor predictions
    """
    ys, acc = run_lssi_on_test_set(models[0], 1)
    ys, don = run_lssi_on_test_set(models[1], 2)
    return ys, acc, don


@permacache(
    "modular_splicing/lssi/analyze/_lssi_coverage_stats_3",
    key_function=dict(models=stable_hash),
)
def _lssi_coverage_stats(models):
    """
    Return the coverage stats for the LSSI models
    """
    ys, acc, don = run_both_lssi_on_test_set(models)
    acc = acc > LSSI_MODEL_THRESH
    don = don > LSSI_MODEL_THRESH
    splice_sites_covered = {3: acc[ys == 1].mean(), 5: don[ys == 2].mean()}
    bases_covered = {3: acc.mean(), 5: don.mean()}
    return splice_sites_covered, bases_covered


def lssi_coverage_stats():
    return _lssi_coverage_stats([x.model for x in LSSI])


def topk(yps, ys, c):
    """
    Compute the top-k accuracy for the given results and labels
    """
    return ((yps > np.quantile(yps, 1 - (ys == c).mean())) & (ys == c)).sum() / (
        ys == c
    ).sum()


@permacache(
    "modular_splicing/lssi/analyze/lssi_accuracies_2",
    key_function=dict(models=stable_hash),
)
def lssi_accuracies(models):
    """
    Return the coverage stats for the LSSI models
    """
    ys, acc, don = run_both_lssi_on_test_set(models)
    return topk(acc, ys, 1), topk(don, ys, 2)


@permacache(
    "modular_splicing/lssi/analyze/maxent_accuracies_2",
    key_function=dict(models=stable_hash),
)
def maxent_accuracies(models):
    """
    Return the coverage stats for the max-ent models
    """
    _, ys = second_half_of_spliceai_cached()
    acc = run_maxent_on_data(models[0])
    don = run_maxent_on_data(models[1])
    return topk(acc, ys, 1), topk(don, ys, 2)


def lssi_and_maxent_accuracies():
    return dict(
        lssi=lssi_accuracies([x.model for x in LSSI]),
        maxent=maxent_accuracies([three_prime, five_prime]),
    )
