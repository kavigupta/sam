from permacache import permacache, stable_hash
import torch

from modular_splicing.legacy.hash_model import hash_model

from modular_splicing.utils.run_batched import run_batched


def predict_splicepoints(model, xs, batch_size, **kwargs):
    """
    Predict splicepoint probabilities for the given sequences, using the given model.

    Parameters
    ----------
    model : a model
    xs : (N, L + model.cl, 4)
        The sequences to predict splicepoints for.
    batch_size : int
        The batch size to use. Not taken into account for caching.
    **kwargs
        Additional arguments to pass to the model.

    Returns
    -------
    (N, L, 2)
        The predicted splicepoint probabilities, for each position in each sequence,
            and for each of the two splicepoint types.
        The null class is not included.
    """
    with torch.no_grad():
        model.eval()
        return run_batched(
            lambda x: model(x).softmax(-1)[:, :, 1:], xs, batch_size, **kwargs
        )


@permacache(
    "utils/run_spliceai_model_cached",
    key_function=dict(model=hash_model, xs=stable_hash, batch_size=None, kwargs=None),
)
def predict_splicepoints_cached(model, xs, batch_size, **kwargs):
    """
    Exactly the same as predict_splicepoints, but cached to disk
    """
    return predict_splicepoints(model, xs, batch_size, **kwargs)
