import tqdm.auto as tqdm

import numpy as np
import scipy
from permacache import permacache, stable_hash
from modular_splicing.evaluation.evaluation_criterion import DefaultEvaluationCriterion

from modular_splicing.dataset.generic_dataset import dataset_types
from modular_splicing.evaluation.run_model import run_model_on_all_batches
from modular_splicing.utils.construct import construct
from .indices_to_graph import generate_closeness_matrix


def fuzzy_topk(y, yp, *, delta):
    yp = yp > np.quantile(yp, 1 - y.mean())
    [yp] = np.where(yp)
    [y] = np.where(y)
    if y.size == 0:
        return np.nan
    if yp.size == 0:
        return 0
    matching = scipy.sparse.csgraph.maximum_bipartite_matching(
        generate_closeness_matrix(y, yp, delta=delta)
    )
    return (matching >= 0).mean()


def fuzzy_topks(yp, y, *, delta_max):
    return np.array(
        [fuzzy_topk(y, yp, delta=delta) for delta in np.arange(delta_max + 1)]
    )


@permacache("metric/fuzzy_topk/all_fuzzy_topks", key_function=dict(model=stable_hash))
def all_fuzzy_topks(model, dset_spec, *, delta_max, limit=float("inf")):
    dset = construct(dataset_types(), dset_spec)
    outputs = run_model_on_all_batches(
        model,
        dset,
        limit=limit,
        bs=32,
        pbar=tqdm.tqdm,
        evaluation_criterion=DefaultEvaluationCriterion(),
    )
    ytrues = [output["trues"] for output in outputs]
    ypreds = [output["preds"] for output in outputs]

    ytrues, ypreds = np.concatenate(ytrues, axis=1), np.concatenate(ypreds, axis=1)
    return np.array(
        [
            fuzzy_topks(ypreds[c], ytrues[c], delta_max=delta_max)
            for c in range(ytrues.shape[0])
        ]
    )
