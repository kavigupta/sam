from permacache import permacache, stable_hash
import tqdm.auto as tqdm
from modular_splicing.legacy.remapping_pickle import permacache_with_remapping_pickle
from modular_splicing.utils.arrays import Sparse

from modular_splicing.utils.run_batched import run_batched


def _predict_motifs_raw(m, xs, bs):
    """
    Predict the motifs for the given sequences, using the given model,
    in batches of size bs.

    Uncached.
    """
    return run_batched(
        lambda x: m(x, only_motifs=True)["post_sparse_motifs_only"],
        xs,
        bs,
        pbar=tqdm.tqdm,
    )


@permacache(
    "modular_splicing/evaluation/predict_motifs/predict_motifs",
    dict(m=stable_hash, xs=stable_hash, bs=None),
)
def predict_motifs(m, xs, bs=8):
    """
    Predict the motifs for the given sequences, using the given model,

    Parameters
    ----------
    m : a model
    xs : (N, L, 4)
        The sequences to predict motifs for.
    bs : int
        The batch size to use. Not taken into account for caching.

    Returns
    -------
    (N, L, M)
        The predicted motifs.
    """
    return _predict_motifs_raw(m, xs, bs)


@permacache_with_remapping_pickle(
    "modular_splicing/evaluation/predict_motifs/predict_motifs_sparse",
    dict(m=stable_hash, xs=stable_hash, bs=None),
)
def predict_motifs_binarized_sparse(m, xs, bs=8):
    """
    Like predict_motifs, but returns a Sparse object instead of a dense array

    Also returns 1/0 rather than the prediction scores.
    """
    return Sparse.of(_predict_motifs_raw(m, xs, bs) != 0)
