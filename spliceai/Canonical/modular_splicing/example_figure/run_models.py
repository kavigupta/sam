from permacache import permacache, stable_hash
import tqdm.auto as tqdm

import torch
import numpy as np

from modular_splicing.data_for_experiments.standardized_sample import (
    calibrate_thresholds_on_standardized_sample,
)
from .data import dataset_for_species


def compute_thresholds(model, species):
    _, path, kwargs = dataset_for_species(species)
    return calibrate_thresholds_on_standardized_sample(
        model,
        path=path,
        amount=5000,
        **kwargs,
    )


@permacache(
    "modular_splicing/figure/main_example_figure/attach_motif_predictions_3",
    key_function=dict(mods=stable_hash, exons=stable_hash),
)
def attach_motif_predictions(mods, exons):
    """
    Attach motif predictions to the exons, using the models in `mods`.

    `mods` is a dictionary mapping model names to models.

    Adds a key to each exon with the name of the model, and the value is a
    dictionary with keys:
        - "mot": the motif predictions
        - "res": the splicepoint predictions
    """
    new_exons = []
    for ex in tqdm.tqdm(exons):
        extra = {}
        for key, mod in mods.items():
            with torch.no_grad():
                x_torch = torch.tensor([ex["x"]]).cuda()
                out = mod(x_torch, collect_intermediates=True)

                res = out["output"].log_softmax(-1)[0, :, 1:].cpu().numpy()
                mot = (
                    out["post_sparse_motifs_only"]
                    .cpu()
                    .numpy()[0, mod.cl // 2 : -mod.cl // 2, :79]
                )
            extra[key] = dict(res=res, mot=mot)
        new_exons.append({**ex, **extra})
    return new_exons


def limit(exons, to=2000):
    if len(exons) > to:
        exons = [
            exons[i]
            for i in np.random.RandomState(0).choice(len(exons), to, replace=False)
        ]
        print(f"Arbitrarily limited: {len(exons)}")
    return exons
