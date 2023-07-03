import numpy as np
import torch
import tqdm.auto as tqdm

from permacache import permacache, stable_hash, drop_if_equal
from modular_splicing.evaluation.predict_splicing import predict_splicepoints_cached

from modular_splicing.models_for_testing.load_model_for_testing import (
    achieved_target_acc,
)
from modular_splicing.utils.io import load_model
from modular_splicing.legacy.hash_model import hash_model
from modular_splicing.utils.run_batched import run_batched


def run_motifs(path, xs, ys, count, get_yps=True):
    print(path)
    m = load_sparsest(path).eval()
    return run_motifs_using_model(m, xs, ys, count, get_yps=get_yps)


def load_sparsest(name, kth=1):
    path = f"model/{name}"
    step = sparsest(path, kth=kth)
    _, m = load_model(path, step=step)
    return m


def sparsest(path, kth=1):
    steps, _ = achieved_target_acc(path)
    return steps[-kth]


def run_motifs_using_model(m, xs, ys, count, get_yps=True, *, motif_indices=None):
    mots = run_all_motifs_for_model(m, xs, motif_indices=motif_indices)
    if motif_indices is None:
        assert mots.shape[-1] in {count, count + 1}
    pre_adj = mots[:, 1, :, :count]
    adj_direct = pre_adj > np.median(pre_adj[pre_adj != 0])
    return dict(
        yps=calibrate(predict_splicepoints_cached(m, xs, 4, pbar=tqdm.tqdm), ys)
        if get_yps
        else None,
        motifs=mots[:, 0, :, :count] != 0,
        motifs_pre_adjustment=pre_adj != 0,
        motifs_adjustment_direct=adj_direct,
    )


@permacache(
    "notebooks/adjustment-model-analysis/run_all_motifs_for_model_2",
    dict(m=hash_model, xs=stable_hash, motif_indices=drop_if_equal(None)),
)
def run_all_motifs_for_model(m, xs, *, motif_indices=None):
    return run_batched(
        lambda x: run_model(m, x, motif_indices=motif_indices),
        xs,
        8,
        pbar=tqdm.tqdm,
    )


def run_model(m, x, *, motif_indices=None):
    y = m(x, only_motifs=True)
    motifs = y["post_sparse_motifs_only"] != 0
    motifs = torch.stack([motifs, y.get("pre_adjustment_motifs", motifs)], axis=1)
    motifs = motifs[:, :, :, motif_indices] if motif_indices is not None else motifs
    return motifs


def calibrate(yps, ys):
    return yps > np.percentile(yps, 100 * (1 - ys.mean()))
