import copy
from functools import lru_cache
import numpy as np
import pandas as pd

from permacache import permacache

import tqdm.auto as tqdm

from modular_splicing.models_for_testing.list import binarized_models
from modular_splicing.evaluation.run_evaluation import evaluate_model_on_data
from modular_splicing.legacy.hash_model import hash_model
from modular_splicing.evaluation import standard_e2e_eval


@lru_cache(None)
def table_of_combinations():
    """
    Evaluate combinations of binarized models. Do not evaluate AMs on other AMs.
    """
    models = binarized_models()
    model_names = [m.name for m in models]
    models = {m.name: m.model for m in models}
    result = evaluate_all_combinations(
        models_for_motifs=models,
        models_for_downstream=models,
        filter=lambda x, y: x == y or y.startswith("FM"),
    )
    motif_models = [f"M={i}" for i in model_names]
    downstream_models = [f"D={i}" for i in model_names]
    result_df = (
        pd.DataFrame(result, index=motif_models, columns=downstream_models) * 100
    )
    result_df = result_df.loc[~np.isnan(result_df.T).all(), ~np.isnan(result_df).all()]
    return result_df


def evaluate_all_combinations(*, models_for_motifs, models_for_downstream, filter):
    """
    Evaluate all combinations of models_for_motifs and models_for_downstream, return the result as
        a matrix, where rows correspond to motif models and columns to downstream models.

    Any pair of names where filter(name1, name2) is False is not evaluated, and instead a NaN is
        returned.
    """
    flat = evaluate_all_combinations_flat(
        m1s=models_for_motifs,
        m2s=models_for_downstream,
        filter=filter,
        limit=float("inf"),
    )
    result = []

    for n1 in models_for_motifs:
        result.append([])
        for n2 in models_for_downstream:
            result[-1].append(flat.get((n1, n2), np.nan))
    return result


def evaluate_all_combinations_flat(m1s, m2s, filter, limit):
    """
    Evaluate all combinations of m1s and m2s, return the result as a dict mapping (name1, name2) to
        the result.

    Any pair of names where filter(name1, name2) is False is not evaluated, and instead a NaN is
        returned.
    """
    result = {
        (n1, n2): np.mean(
            evaluate_model_combination(m1s[n1], m2s[n2], limit=limit, bs=10)
        )
        for (n1, n2) in tqdm.tqdm(
            [(n1, n2) for n1 in m1s for n2 in m2s if filter(n1, n2)]
        )
    }
    return result


@permacache(
    "notebooks/binarized-adjusted-motif-models/evaluate_model_combination_3",
    key_function=dict(
        model_for_motifs=hash_model, model_for_downstream=hash_model, bs=None
    ),
)
def evaluate_model_combination(
    model_for_motifs, model_for_downstream, limit, bs, path="./dataset_test_0.h5"
):
    """
    Evaluate the given model combination. Return an evaluation result.
    """
    assert path == "./dataset_test_0.h5", "Only allow running this on the test data"
    if model_for_motifs is None or model_for_downstream is None:
        return np.nan
    model_for_downstream = combine_models(model_for_motifs, model_for_downstream)

    deval = get_deval(path=path, cl=model_for_downstream.cl)
    return evaluate_model_on_data(
        model_for_downstream, deval, limit=limit, bs=bs, pbar=tqdm.tqdm
    )


def combine_models(model_for_motifs, model_for_downstream):
    """
    Combine the two models together, taking the motif model as the first half and the downstream
        model as the second half.

    Does not mutate either model.
    """
    model_for_downstream = copy.deepcopy(model_for_downstream)
    model_for_downstream.motif_model = model_for_motifs.motif_model
    model_for_downstream.sparsity_enforcer = model_for_motifs.sparsity_enforcer
    return model_for_downstream


def get_deval(*, path, cl):
    """
    Get the evaluation dataset (just the second half of spliceai).
    """
    return standard_e2e_eval.test_data(
        dict(
            type="H5Dataset",
            sl=5000,
            datapoint_extractor_spec=dict(type="BasicDatapointExtractor"),
            post_processor_spec=dict(type="IdentityPostProcessor"),
        ),
        cl=cl,
        data_path=path,
    )
