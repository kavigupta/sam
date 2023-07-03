import os

import tqdm.auto as tqdm
import numpy as np
import torch.nn as nn

from permacache import permacache, drop_if_equal, stable_hash


from modular_splicing.evaluation.evaluation_criterion import DefaultEvaluationCriterion
from modular_splicing.utils.construct import construct
from modular_splicing.evaluation.run_evaluation import evaluate_model_on_data


@permacache(
    "utils/evaluate_checkpoint_on_second_half_of_spliceai",
    key_function=dict(
        path=os.path.abspath,
        bs=None,
        pbar=None,
        data_spec=drop_if_equal(None),
        evaluation_criterion=drop_if_equal(DefaultEvaluationCriterion()),
        force_cl=drop_if_equal(None),
        data_path=drop_if_equal("dataset_test_0.h5"),
        split=drop_if_equal("test"),
    ),
)
def evaluate_on_checkpoint(
    *,
    path,
    step,
    limit,
    bs,
    pbar,
    evaluation_criterion=DefaultEvaluationCriterion(),
    data_spec,
    force_cl=None,
    data_path="dataset_test_0.h5",
    split="test",
):
    """
    Evaluate the given model on the given checkpoint. See `evaluate_on_model`
        for more details on the parameters, this function just wraps the loading of the checkpoint
        in order to have a faster cache.
    """
    from modular_splicing.utils.io import load_model

    _, model = load_model(path, step)
    assert model is not None, f"Could not load model from {path} at step {step}"

    if not isinstance(model, nn.Module):
        model = model.model

    return evaluate_on_model(
        model=model,
        limit=limit,
        bs=bs,
        pbar=pbar,
        evaluation_criterion=evaluation_criterion,
        data_spec=data_spec,
        force_cl=force_cl,
        data_path=data_path,
        split=split,
    )


@permacache(
    "modular_splicing/evaluation/standard_e2e_eval/evaluate_on_model",
    key_function=dict(
        model=stable_hash,
        bs=None,
        pbar=None,
        data_spec=drop_if_equal(None),
        evaluation_criterion=drop_if_equal(DefaultEvaluationCriterion()),
        force_cl=drop_if_equal(None),
        data_path=drop_if_equal("dataset_test_0.h5"),
        split=drop_if_equal("test"),
    ),
)
def evaluate_on_model(
    *,
    model,
    limit,
    bs,
    pbar,
    evaluation_criterion=DefaultEvaluationCriterion(),
    data_spec,
    force_cl=None,
    data_path="dataset_test_0.h5",
    split="test",
):
    """
    Evaluate the given model on the second half of spliceai.

    Parameters
    ----------
    model: nn.Module
        The model to evaluate.
    limit: int
        The maximum number of samples to evaluate on.
    bs: int
        The batch size to use.
    pbar: tqdm.tqdm
        A progress bar to use.
    evaluation_criterion: EvaluationCriterion
        The criterion to use for evaluation.
    data_spec: dict
        The data spec to use for the data.
    force_cl: int
        If not None, force the data's context length to be this value.
        Otherwise, use the model's context length.
    """
    assert split in ["test", "val"]
    cl = force_cl if force_cl is not None else model.cl
    data = test_data(data_spec, cl, data_path=data_path, skip_for_test=split == "test")

    return evaluate_model_on_data(
        model,
        data,
        limit=min(limit, len(data) // 2 if split == "val" else limit),
        bs=bs,
        pbar=lambda *args, **kwargs: pbar(*args, desc=f"<model>", **kwargs),
        evaluation_criterion=evaluation_criterion,
    )


def test_data(data_spec, cl, *, data_path="dataset_test_0.h5", skip_for_test=True):
    """
    Data to test things on. Consists of the last 45% of the SpliceAI test set.

    If skip_for_test is False, we just return the full test set. This is useful in case we want to
        use validation accuracy.
    """

    from modular_splicing.dataset.generic_dataset import dataset_types

    iterator_spec = dict(type="FastIter", shuffler_spec=dict(type="DoNotShuffle"))

    if skip_for_test:
        iterator_spec = dict(
            type="SkipFirst",
            iterator_spec=iterator_spec,
            skip_first_frac=0.55,
        )

    return construct(
        dataset_types(),
        data_spec,
        path=data_path,
        cl=cl,
        cl_max=10_000,
        # cl_max=400,
        iterator_spec=iterator_spec,
    )


def evaluate_model_with_step(
    m,
    data_spec=dict(
        type="H5Dataset",
        sl=5000,
        datapoint_extractor_spec=dict(type="BasicDatapointExtractor"),
        post_processor_spec=dict(type="IdentityPostProcessor"),
    ),
):
    """
    Evaluate a ModelForTesting object.
    """
    result = evaluate_on_checkpoint(
        path=m.path,
        step=m.step,
        limit=float("inf"),
        bs=32,
        pbar=tqdm.tqdm,
        data_spec=data_spec,
    )
    return result
