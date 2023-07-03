import numpy as np
import torch


def run_batched(m, x, bs, pbar=lambda x: x):
    """
    Run m on x in batches of size bs, and concatenate the results.

    If x is a dict, treat it as a dict of arrays, each of which is batched
        together. This requires that the length of each array is the same.

    If m returns a dict, treat it as a dict of arrays, each of which is
        concatenated together. This requires that the length of each array is
        the same.

    Arguments
    ---------
    m : function
        A function which takes a batch of data and returns a batch of data.
    x : array or dict of arrays
        The data to be batched.
    bs : int
        The batch size.
    pbar : function
        A function which takes an iterable and returns an iterable. This is
        used to wrap the iterable over the batches of data to produce a pbar.

    Returns
    -------
    array or dict of arrays
        The result of running m on x in batches of size bs.
    """
    if isinstance(x, dict):
        [size] = set(x.shape[0] for x in x.values())
    else:
        size = x.shape[0]
    assert size != 0

    with torch.no_grad():
        ys = []
        for i in pbar(range((size + bs - 1) // bs)):
            select = lambda x: torch.tensor(x[i * bs : (i + 1) * bs]).cuda()
            x_to_use = (
                {k: select(x[k]) for k in x} if isinstance(x, dict) else select(x)
            )
            y = m(x_to_use)
            y = to_numpy(y)
            ys.append(y)
        result = concatenate_all(ys)
        return result


def concatenate_all(ys):
    """
    Concatenate a list of arrays or dicts of arrays.

    Arguments
    ---------
    ys : list of arrays or dicts of arrays
        The arrays to be concatenated.

    Returns
    -------
    array or dict of arrays
    """
    if isinstance(ys[0], dict):
        assert all(y.keys() == ys[0].keys() for y in ys)
        result = {k: np.concatenate([y[k] for y in ys]) for k in ys[0]}
    else:
        result = np.concatenate(ys)
    return result


def to_numpy(y):
    """
    Convert the following tensor or dict of tensors to numpy arrays.
    """
    if isinstance(y, dict):
        y = {k: y[k].cpu().numpy() for k in y}
    else:
        y = y.cpu().numpy()
    return y
