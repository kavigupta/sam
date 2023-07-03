from collections import defaultdict
import itertools
import os
import warnings

import numpy as np
import torch

from permacache import permacache, stable_hash
from more_itertools import chunked

from modular_splicing.dataset.basic_dataset import basic_dataset


@permacache(
    "analysis/extract_psams",
    key_function=dict(
        m=stable_hash,
        data_path=os.path.abspath,
        pbar=None,
        bs=None,
    ),
)
def extract_psams(m, data_path, *, mcl, count, bs, pbar=lambda x, **kwargs: x):
    by_motif = motif_binding_sites(m, data_path, mcl=mcl, count=count, bs=bs, pbar=pbar)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        psams = [
            np.nan_to_num(np.eye(4)[np.array(mot)].mean(0), nan=1) for mot in by_motif
        ]
    return np.array(psams)


def motif_binding_sites(m, data_path, *, mcl, count, bs, pbar=lambda x, **kwargs: x):
    data = basic_dataset(data_path, mcl, 10_000, sl=5000)
    by_motif = defaultdict(list)
    motif_indices = m.sparsity_enforcer.motif_index()
    xvals = (xs for xs, _ in data)
    for xs in pbar(
        chunked(itertools.islice(xvals, count), bs), total=(count + bs - 1) // bs
    ):
        xs = np.array(xs)
        xseq = xs.argmax(-1)
        with torch.no_grad():
            motifs = (
                m(torch.tensor(xs).cuda(), only_motifs=True)["post_sparse_motifs_only"]
                .cpu()
                .numpy()
            )
        batch, starts, motif_idxs = np.where(motifs[:, mcl // 2 : -mcl // 2])
        for b, seq, mot in zip(batch, starts, motif_idxs):
            mot = motif_indices[mot]
            by_motif[mot].append(xseq[b, seq : seq + mcl + 1])
    results = []
    for i in range(1 + max(by_motif)):
        if len(by_motif[i]):
            results.append(np.array(by_motif[i], dtype=np.int64))
        else:
            results.append(np.zeros((0, mcl + 1), dtype=np.int64))
    return results


def extract_psams_from_motifs(xs, motifs, mcl, pbar=lambda x: x):
    psams = []
    for motif_idx in pbar(range(motifs.shape[-1])):
        batch_idxs, seq_idxs = np.where(motifs[:, :, motif_idx])
        mask = seq_idxs < motifs.shape[1] - (mcl + 1) // 2
        mask &= seq_idxs > mcl // 2
        batch_idxs, seq_idxs = batch_idxs[mask], seq_idxs[mask]
        psam = []
        for off in range(-mcl // 2, 1 + mcl // 2):
            pos = xs[batch_idxs, seq_idxs + off]
            if pos.shape[0] == 0:
                psam.append([np.nan] * 4)
            else:
                psam.append(pos.mean(0))
        psams.append(psam)
    return np.array(psams)
