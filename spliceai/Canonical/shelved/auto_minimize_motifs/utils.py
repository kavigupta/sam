import os

from permacache import permacache, drop_if_equal

import torch
import numpy as np

from modular_splicing.utils.io import model_steps, load_model


def amm_get_step_with_dropped_motifs(path, n_dropped_motifs):
    steps = model_steps(path)
    num_dropped_motifs = np.array(
        [len(dropped_motifs(path, step, named=True)) for step in steps]
    )
    [idxs] = np.where(num_dropped_motifs == n_dropped_motifs)
    idx = idxs.max()
    return steps[idx]


@permacache(
    "spliceai/statistics/dropped_motif",
    key_function=dict(path=os.path.abspath, named=drop_if_equal(True)),
)
def dropped_motifs(path, step, named=False):
    _, m = load_model(path, step, map_location=torch.device("cpu"))
    original_m = m
    if not hasattr(m, "sparse_layer"):
        m = m.model
    dropped = getattr(
        m.sparse_layer, "dropped", getattr(m.sparse_layer, "dropped_motifs", [])
    )
    if named:
        dropped = [original_m.names[i] for i in dropped]
    return dropped
