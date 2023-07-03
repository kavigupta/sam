import os

import tqdm.auto as tqdm

from modular_splicing.example_figure.find_example_exon import find_both_exons
from modular_splicing.example_figure.renderer import main_example_figure
from modular_splicing.example_figure.replicates import attach_alternate


def produce_images(species, lssi, *, fms, ams, range_multiplier=1, require_tra2a):
    """
    Produce images for the given species.
    """
    assert len(fms) == len(ams)
    assert len(fms) > 0
    pos, neg = find_both_exons(
        lssi=lssi,
        fm=fms[0],
        am=ams[0],
        species=species,
        range_multiplier=range_multiplier,
        require_tra2a=require_tra2a,
    )
    location = f"{species}_log"
    if require_tra2a:
        location += "_filtered_for_tra2a"
    render_several(
        pos, neg, location, species=species, range_multiplier=range_multiplier
    )
    for i, (fm, am) in tqdm.tqdm(
        list(enumerate(zip(fms[1:], ams[1:]))), desc="Alternates"
    ):
        pos_i, neg_i = [
            attach_alternate(
                dict(FM=fm, AM=am), exons, range_multiplier=range_multiplier
            )
            for exons in (pos, neg)
        ]
        render_several(
            pos_i,
            neg_i,
            location,
            species=species,
            range_multiplier=range_multiplier,
            suffix=f"_replicate_{i + 2}",
        )


def render_several(pos, neg, folder, *, suffix="", species, **kwargs):
    try:
        os.makedirs(f"output-csvs/main-figure/{folder}")
    except FileExistsError:
        pass
    for i, ex in enumerate(pos[:25]):
        main_example_figure(
            ex, f"{folder}/positive_{i}{suffix}", species=species, **kwargs
        )
    for i, ex in enumerate(neg[:25]):
        main_example_figure(
            ex, f"{folder}/negative_{i}{suffix}", species=species, **kwargs
        )
