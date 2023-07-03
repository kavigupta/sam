import numpy as np

from modular_splicing.evaluation.evaluation_criterion import DefaultEvaluationCriterion
from .run_model import run_model_on_all_batches


def top_kl_accuracy(y_true, y_pred, top_length=1):
    """
    Compute topk-l accuracy. By default, l=1.

    Parameters
    ----------
    y_true : array-like, shape = [n_samples], dtype=bool or float
        True labels. If dtype is float, then we use the threshold 0.5
        to convert to boolean.
    y_pred : array-like, shape = [n_samples], dtype=float
        Predicted scores.
    top_length : int
        The multiplier in top-kl accuracy.

    Returns
    -------
    accuracy : float
        Top-k accuracy.
    """

    idx_true = np.nonzero(y_true > 0.5)[0]

    kl = int(top_length * len(idx_true))

    idx_pred = np.argpartition(y_pred, -kl)[-kl:]

    intersection_size = np.size(np.intersect1d(idx_true, idx_pred))
    max_intersection_size = min(len(idx_pred), len(idx_true))
    if max_intersection_size == 0:
        return np.nan
    else:
        return intersection_size / max_intersection_size


def accuracy_from_batches(outputs):
    """
    Compute accuracies from the given list of outputs.

    Parameters
    ----------
    outputs : list of (y_true, y_pred)
        The list of outputs. Each should be an output of `run_model_on_single_batch`.

    Returns
    -------
    accuracies : list of float
        The list of top-k accuracies, one per channel.
    """
    assert all(output.keys() == {"trues", "preds"} for output in outputs)
    ytrues = [output["trues"] for output in outputs]
    ypreds = [output["preds"] for output in outputs]

    # each is now of shape (C, N), where B is the batch index and C is the number of channels.
    ytrues, ypreds = np.concatenate(ytrues, axis=1), np.concatenate(ypreds, axis=1)

    by_c = []
    for c in range(1, len(ytrues) + 1):
        yt = ytrues[c - 1]
        yp = ypreds[c - 1]
        by_c.append(top_kl_accuracy(yt, yp))
    return by_c


def evaluate_model_on_data(
    m,
    d,
    limit=float("inf"),
    bs=32,
    pbar=lambda x, **_: x,
    model_kwargs={},
    *,
    evaluation_criterion=DefaultEvaluationCriterion(),
):
    """
    Evaluate the given model on the given data.

    Parameters
    ----------
    m : torch.nn.Module
        The model to evaluate.
    d : dataset.GenericDataset
        The dataset to evaluate on.
    limit : int
        The maximum number of elements to evaluate on.
        Can be slightly higher, by as much as the batch size.
    bs : int
        The batch size.
    pbar : function
        A function that takes an iterable and returns an iterable with a progress bar.
    model_kwargs : dict
        Keyword arguments to pass to the model.
    evaluation_criterion : EvaluationCriterion
        The criterion to use for evaluation.

    Returns
    -------
    results : list
        The list of top-k results, one per channel.
    """
    outputs = run_model_on_all_batches(
        m,
        d,
        limit=limit,
        bs=bs,
        pbar=pbar,
        model_kwargs=model_kwargs,
        evaluation_criterion=evaluation_criterion,
    )

    return accuracy_from_batches(outputs)
