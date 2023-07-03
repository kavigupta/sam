import copy

from permacache import permacache, stable_hash

from .run_models import attach_motif_predictions
from .find_example_exon import attach_effects, get_default_effect_kwargs


@permacache(
    "modular_splicing/example_figure/replicates/attach_alternate",
    key_function=dict(models=stable_hash, exons=stable_hash),
)
def attach_alternate(models, exons, *, range_multiplier):
    """
    Attach the alternate model predictions to the exons.
    """
    assert models.keys() == {"FM", "AM"}
    exons = copy.deepcopy(exons)
    for exon in exons:
        del exon["FM"], exon["AM"]
    exons = attach_motif_predictions(models, exons)
    exons = attach_effects(
        models,
        exons,
        **get_default_effect_kwargs(range_multiplier),
    )
    return exons
