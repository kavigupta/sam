import numpy as np
import pandas as pd
import tqdm.auto as tqdm

from .standard_e2e_eval import evaluate_on_checkpoint


def evaluate_model_series(
    model_series,
    data_spec,
    *,
    split,
    include_path=False,
    data_path="dataset_test_0.h5",
):
    """
    Evaluate a series of models, with the same data spec.

    Arguments
    ---------
    model_series: `EndToEndModelsForTesting` instance. The series of models to evaluate.
    data_spec: `dict`. The data spec to use for evaluation.

    Returns
    -------
    `dict`. The results of evaluation, mapping from seed to the mean evaluation result.
    """
    result = {
        model.seed: evaluate_individual_model(
            model, data_spec, split=split, data_path=data_path
        )
        for model in model_series.non_binarized_models()
    }
    if include_path:
        result = {"path": model_series.path_prefix, **result}
    return result


def evaluate_individual_model(
    model, data_spec, *, split="test", data_path="dataset_test_0.h5"
):
    """
    Evaluate a single model, with the given data spec.

    Arguments
    ---------
    model: `ModelForTesting` instance. The model to evaluate.
    data_spec: `dict`. The data spec to use for evaluation.

    Returns
    -------
    `float`. The mean evaluation result.
    """
    try:
        s = model.step
    except ValueError:
        return np.nan
    except FileNotFoundError:
        return np.nan
    return np.mean(
        evaluate_on_checkpoint(
            path=model.path,
            step=s,
            limit=float("inf"),
            bs=32,
            pbar=tqdm.tqdm,
            data_spec=data_spec,
            split=split,
            data_path=data_path,
        )
    )


def evaluate_all_series(
    *model_series_and_data,
    split="test",
    include_path=False,
    data_path="dataset_test_0.h5",
):
    """
    Evaluate the given serieses of models, with the given data specs.

    Arguments
    ---------
    model_series_and_data: `tuple` of `tuple`s. Each tuple is a model series and a data spec.

    Returns
    -------
    `dict`. The results of evaluation, mapping from series name to seed to the mean evaluation result.
    """
    return {
        series.name: evaluate_model_series(
            series,
            data_spec,
            split=split,
            include_path=include_path,
            data_path=data_path,
        )
        for series, data_spec in model_series_and_data
    }
