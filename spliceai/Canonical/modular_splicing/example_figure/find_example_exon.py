import copy

from permacache import permacache, stable_hash, drop_if_equal

import numpy as np
import tqdm.auto as tqdm

from modular_splicing.dataset.datafile_object import SpliceAIDatafile
from modular_splicing.dataset.dataset_aligner import DatasetAligner
from modular_splicing.example_figure.run_models import compute_thresholds
from .correctness import (
    KEY_BY_NAME,
    ensure_several_correctness_values,
)
from .extract_exons import get_exons

from .effect import compute_effect
from modular_splicing.motif_names import get_motif_names
from .data import dataset_for_species

LSSI_FILTER_MULTIPLIER = 2


@permacache(
    "modular_splicing/example_figure/find_example_exon/find_both_exons_16",
    key_function=dict(
        fm=stable_hash, am=stable_hash, require_tra2a=drop_if_equal(True)
    ),
)
def find_both_exons(*, lssi, fm, am, species, range_multiplier=1, require_tra2a=True):
    """
    Finds both positive and negative examples
    """
    common_params = dict(
        flank_size=100,
        min_len=50,
        max_len=250,
        lssi_filter_multiplier=LSSI_FILTER_MULTIPLIER,
    )
    ex_pos = find_example_exon(
        lssi=lssi,
        fm=fm,
        am=am,
        species=species,
        **common_params,
        effect_kwargs=get_default_effect_kwargs(range_multiplier),
        require_tra2a=require_tra2a,
    )
    ex_neg = find_negative_example_exon(
        lssi=lssi,
        fm=fm,
        am=am,
        species=species,
        **common_params,
        effect_kwargs=get_default_effect_kwargs(range_multiplier),
    )
    return ex_pos, ex_neg


def get_default_effect_kwargs(range_multiplier):
    return dict(
        effect_thresh=np.log(1.5),
        kind_of_effect="log",
        minimal_probability_requirement_for_effect=np.exp(-5 * range_multiplier),
    )


def find_example_exon(
    *,
    lssi,
    fm,
    am,
    flank_size,
    min_len,
    max_len,
    species,
    lssi_filter_multiplier,
    effect_kwargs,
    require_tra2a=True,
):
    """
    Finds an exon

    - that is between min_len and max_len bases long
    - has flanking introns that are each at least flank_size bases long
    - LSSI and FM are incorrect. LSSI is required to predict
        at least one false positive and one false negative.
    - AM is correct
    - TRA2A appears in the sequence and has an effect on splicing
    """
    assert am.cl == fm.cl
    exons = appropriately_sized_exons(
        am.cl,
        flank=flank_size,
        min_len=min_len,
        max_len=max_len,
        lssi_model=lssi,
        species=species,
        use_prediction=False,
        lssi_filter_multiplier=lssi_filter_multiplier,
    )
    print(f"Appropriately sized: {len(exons)}")
    exons = ensure_several_correctness_values(
        exons,
        dict(LSSI=lssi, FM=fm, AM=am),
        ("LSSI", "incorrect_without_subset", lssi_filter_multiplier),
        ("AM", "correct", 1),
        ("FM", "incorrect", 1),
        # ("FM", "incorrect_without_subset", 1),
        species=species,
    )
    return clean_up_exons(
        exons,
        fm=fm,
        am=am,
        lssi=lssi,
        species=species,
        effect_kwargs=effect_kwargs,
        require_tra2a=require_tra2a,
    )


def find_negative_example_exon(
    *,
    lssi,
    fm,
    am,
    flank_size,
    min_len,
    max_len,
    species,
    lssi_filter_multiplier,
    effect_kwargs,
):
    """
    Finds a fake exon

    - that is predicted by LSSI (with a multiplier)
    - does not exist (and there is no splicing on the flanks or in the "exon")
    - FM is incorrect (just means it predicts something, not necessarily the false exon)
    - AM is correct
    """
    assert am.cl == fm.cl
    cl = am.cl
    exons = appropriately_sized_exons(
        am.cl,
        flank=flank_size,
        min_len=min_len,
        max_len=max_len,
        lssi_model=lssi,
        species=species,
        use_prediction=True,
        lssi_filter_multiplier=lssi_filter_multiplier,
    )
    print(f"Appropriately sized: {len(exons)}")
    exons = [x for x in exons if (x["y"] == 0).all()]
    print(f"No actual splicepoints present: {len(exons)}")
    exons = ensure_several_correctness_values(
        exons,
        dict(LSSI=lssi, FM=fm, AM=am),
        ("FM", "incorrect", 1),
        ("AM", "correct", 1),
        species=species,
    )
    return clean_up_exons(
        exons, fm=fm, am=am, lssi=lssi, species=species, effect_kwargs=effect_kwargs
    )


def clean_up_exons(exons, *, fm, am, lssi, species, effect_kwargs, require_tra2a=False):
    exons = copy.deepcopy(exons)
    models = dict(LSSI=lssi, FM=fm, AM=am)
    thresholds = {k: compute_thresholds(models[k], species) for k in models}
    print(f"Thresholds: {thresholds}")
    for k in models:
        for ex in exons:
            arr = KEY_BY_NAME[k](ex)
            # just suppress low LSSI predictions
            # since we are rescaling we want to ensure that
            # it ends up being low after rescaling
            # and semantically it is -inf when it's below the threshold
            arr[arr < -10] = -np.inf
            arr /= -np.log(thresholds[k])

    exons = attach_effects({"FM": fm, "AM": am}, exons, **effect_kwargs)
    if require_tra2a and species == "human":
        tra2a_idx = get_motif_names("rbns").index("TRA2A")
        print(f"TRA2A idx: {tra2a_idx}")
        exons = [
            ex
            for ex in exons
            if any(x["mot_id"] == tra2a_idx for x in ex["AM"]["effects"])
        ]
        print(f"Contains TRA2A: {len(exons)}")
    exons = add_gene_position_to_each(exons, species=species)
    print(f"Positive strand: {len(exons)}")
    return exons


def appropriately_sized_exons(cl, flank, min_len, max_len, **kwargs):
    """
    Get sequences of size between min_len and max_len, where there aren't any
    splices on the flanks of the exon
    """
    exons = list(get_exons(flank=flank, cl=cl, **kwargs))
    exons = [ex for ex in exons if (ex["y_to_use"] != 0).sum() == 2]
    exons = [ex for ex in exons if min_len <= ex["e"] - ex["s"] <= max_len]
    return exons


def attach_effects(
    mods,
    exons,
    effect_thresh,
    *,
    kind_of_effect,
    minimal_probability_requirement_for_effect,
):
    """
    Attach effects to the exons, using the models in `mods`.
    """
    new_exons = []
    for ex in tqdm.tqdm(exons, desc="Attach effects"):
        ex = copy.deepcopy(ex)
        for key, mod in mods.items():
            ex[key]["effects"] = list(
                compute_effect(
                    mod,
                    ex,
                    effect_thresh=effect_thresh,
                    cl=mod.cl,
                    kind_of_effect=kind_of_effect,
                    minimal_probability_requirement_for_effect=minimal_probability_requirement_for_effect,
                )
            )
        new_exons.append(ex)
    return new_exons


def add_gene_position(ex, species):
    """
    Annotate the exon with the gene position
    """
    datafile, dataset, _ = dataset_for_species(species)
    aligner = DatasetAligner(dataset, datafile, 5000)
    dfile = SpliceAIDatafile.load(datafile)
    gene_idx, off = aligner.get_gene_idx(*ex["dataset_idx"])
    start = dfile.starts[gene_idx] + ex["startpos"] + off * 5000
    return dict(
        **ex,
        gene_pos=dict(
            gene_idx=dfile.names[gene_idx],
            chrom=dfile.chroms[gene_idx],
            strand=dfile.strands[gene_idx],
            start=start,
        ),
    )


def add_gene_position_to_each(exons, *, species):
    """
    Adds a "gene_pos" field to each exon, which is the position of the exon
    relative to the gene

    Returns only exons on the positive strand
    """
    exons = [add_gene_position(ex, species=species) for ex in exons]
    exons = [ex for ex in exons if ex["gene_pos"]["strand"] == "+"]
    return exons
