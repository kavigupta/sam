import numpy as np
from permacache import permacache, stable_hash

from modular_splicing.data_for_experiments.standardized_sample import (
    standardized_sample_memcache,
    model_motifs_on_standardized_sample,
)


@permacache(
    "modular_splicing/other_experiments/motif_positioning/compute_motif_positioning_relative_to_splice_sites",
    key_function=dict(m=stable_hash),
)
def motif_position_map(m, path, *, amount, cl, indices):
    _, ys = standardized_sample_memcache(path, amount, cl=cl)
    mots = model_motifs_on_standardized_sample(
        model_for_motifs=m,
        indices=indices,
        path=path,
        amount=amount,
        cl=cl,
    )
    overall = []
    for ad in 1, 2:
        batch, seq = np.where(ys == ad)
        results = np.array([mots[b, s : s + cl] for b, s in zip(batch, seq)])
        results = np.mean(results, 0)
        overall.append(results)
    return overall


def motif_position_map_for_series(series, path, *, amount, cl, indices):
    return np.mean(
        [
            motif_position_map(m.model, path, amount=amount, cl=cl, indices=indices)
            for m in series.non_binarized_models()
        ],
        0,
    )
