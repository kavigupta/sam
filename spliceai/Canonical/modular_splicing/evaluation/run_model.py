import torch
from torch.utils.data import DataLoader

from modular_splicing.dataset.for_model import reformat_argument_for_model


def run_model_on_single_batch(*, m, xy, evaluation_criterion, model_kwargs={}):
    """
    Run a model on a single batch.

    m: The model to run.
    xy: The batch to run the model on, should be compatible
        with `reformat_argument_for_model`.
    evaluation_criterion: The EvaluationCriterion to use.
        This criterion will be used to compute the evaluation channels to use
        as well as how to extract the true output from the y value from
        the data (via `for_channel`).
    model_kwargs: Additional kwargs to pass to the model.

    Returns
    -------
    A dictionary with keys `trues` and `preds`.
    Both values are lists of length `len(evaluation_criterion.evaluation_channels(yp)`,
        where every element of these lists is of shape (NL,)
        (we collapse the batch and sequence dimensions).
    """
    arg, y, mask = reformat_argument_for_model(
        m, xy, evaluation_criterion=evaluation_criterion
    )

    if mask.any():
        with torch.no_grad():
            yp = m(arg, **model_kwargs).softmax(-1)

        y, yp = y[mask], yp[mask]
    else:
        y, yp = torch.tensor([], device=y.device, dtype=torch.int64), torch.tensor(
            [], device=y.device
        )

    loop_trues = []
    loop_preds = []
    for c in evaluation_criterion.evaluation_channels(yp):
        tr = evaluation_criterion.for_channel(y, c).cpu().numpy()
        pr = yp[:, c].detach().cpu().numpy()
        loop_trues.append(tr)
        loop_preds.append(pr)
        assert tr.shape == pr.shape, str((tr.shape, pr.shape))
    return dict(trues=loop_trues, preds=loop_preds)


def run_model_on_all_batches(
    m,
    d,
    limit=float("inf"),
    bs=32,
    pbar=lambda x, **_: x,
    model_kwargs={},
    *,
    evaluation_criterion,
):
    """
    Run `run_model` on all batches in `d`.

    Will always try to use an integer number of batches, even if len(limit) % bs != 0.

    E.g., if len(d) == 100 and bs == 32, then the last batch will be of size 32, not 4,
        for a total of 128 elements used.

    Parameters
    ----------
    m: The model to run.
    d: The dataset to run the model on.
    limit: The maximum number of elements to use.
    bs: The batch size to use.
    pbar: A function that takes an iterable and returns an iterable.
        This is used to wrap the batches in a progress bar.
    model_kwargs: Additional kwargs to pass to the model.
    evaluation_criterion: The EvaluationCriterion to use.
        This criterion will be used to compute the evaluation channels to use
        as well as how to extract the true output from the y value from
        the data (via `for_channel`).
    """
    outputs = []
    count = 0
    try:
        set_training_at_end = m.training
        m.eval()
        for xy in pbar(
            DataLoader(d, batch_size=bs), total=(min(len(d), limit) + bs) // bs
        ):
            outputs.append(
                run_model_on_single_batch(
                    m=m,
                    xy=xy,
                    model_kwargs=model_kwargs,
                    evaluation_criterion=evaluation_criterion,
                )
            )
            count += bs
            if count >= limit:
                break
    finally:
        if set_training_at_end:
            m.train()
    outputs = [x for x in outputs if x != ([], [])]
    return outputs
