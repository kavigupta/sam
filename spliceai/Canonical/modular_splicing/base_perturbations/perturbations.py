from more_itertools import chunked
from permacache import permacache, stable_hash

import numpy as np
from modular_splicing.legacy.hash_model import hash_model
from modular_splicing.utils.run_batched import run_batched
from modular_splicing.utils.sequence_utils import all_seqs


def function_on_changed_sequence(
    x, change_window, gather_prediction, pbar=lambda x: x, batch_size=100
):
    """
    Run the given function on all possible kmer perturbations of the given sequence.

    Parameters:
        x: The sequence to use (L, 4)
        change_window: The size of the kmer perturbations to make
        gather_prediction: The function to run on each kmer perturbation (K, N, L, 4) -> (K, N, T)
            where N = 4 ** change_window and K is a batch parameter.

            Should be entirely parallel across K, which is included just for efficiency's sake.
        pbar: A progress bar to use

    Returns:
        A list of (N, T) arrays that is L long.
    """
    count = 4**change_window
    start_points = x.shape[0] - change_window + 1
    seqs = np.array(list(all_seqs(change_window)))
    yps = []
    for i_s in pbar(list(chunked(range(start_points), batch_size))):
        all_modifications = []
        for i in i_s:
            modifications = np.repeat(np.array([x]), count, axis=0)
            modifications[:, i : i + change_window, :] = seqs
            all_modifications.append(modifications)
        yp = gather_prediction(all_modifications)
        yps.extend(yp)

    return yps


@permacache(
    "modular_splicing/base_perturbations/perturbations/probabilities_by_changed_sequence",
    key_function=dict(
        m=hash_model,
        x=stable_hash,
        y=stable_hash,
        pbar=None,
        context=stable_hash,
        bs=None,
    ),
)
def probabilities_by_changed_sequence(
    m, x, y, change_window, pbar=lambda x: x, *, context=None, bs=100
):
    """
    Compute the probabilities of the given model being evaluated on all possible kmer perturbations of the given sequence.

    See function_on_changed_sequence for details on other parameters and output.

    context: if not None, should be a dictionary with keys "left" and "right". Each should be of size (*, 4).
        these will be concatenated to the modified x before being passed to the model.
        This allows you to run models with a greater context length on a narrower perturbation window.
    """

    assert m.cl + 1 == x.shape[0]

    assert y in {1, 2}

    def gather_prediction(x):
        x = np.array(x)
        initial_dims = x.shape[:2]
        x = x.reshape(-1, *x.shape[2:])
        if context is not None:
            left, right = context["left"], context["right"]
            left, right = np.repeat(left[None], x.shape[0], axis=0), np.repeat(
                right[None], x.shape[0], axis=0
            )
            x = np.concatenate([left, x, right], axis=1)
            index = left.shape[1]
        else:
            index = 0
        yp = run_batched(_single(m, index), x, bs)
        yp = yp[:, y]
        yp = yp.reshape(*initial_dims, *yp.shape[1:])
        return yp

    yps = function_on_changed_sequence(
        x, change_window, gather_prediction, pbar, batch_size=bs
    )

    [[orig_pred]] = gather_prediction(np.array([[x]]))

    return orig_pred, np.array(yps)


def _single(m, index):
    """
    Evaluate the model in "single" mode where only a single prediction is made
    """
    return lambda t: m(t).softmax(-1)[:, index, :]
