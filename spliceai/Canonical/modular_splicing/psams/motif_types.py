from collections import defaultdict
import json

from modular_splicing.utils.construct import construct

from .sources import (
    read_rbns_motifs,
    read_rbns_v2p1_motifs,
    read_rbns_functional_motifs,
    read_rbns_v2p1_functional_motifs,
    rbns_rnacompete_motifs_deduplicated,
    rna_compete_motifs_for,
)


def motifs_types():
    """
    Dictionary of motif types.
    """
    return dict(
        rbns=read_rbns_motifs,
        rbns_v2p1=read_rbns_v2p1_motifs,
        rbns_functional=read_rbns_functional_motifs,
        rbns_v2p1_functional=read_rbns_v2p1_functional_motifs,
        rna_compete_motifs_for=rna_compete_motifs_for,
        rna_compete=lambda: rbns_rnacompete_motifs_deduplicated(
            overlap_mode="just-rnacompete"
        ),
        rbrc=lambda: rbns_rnacompete_motifs_deduplicated(overlap_mode="use-rbns"),
        rcrb=lambda: rbns_rnacompete_motifs_deduplicated(overlap_mode="use-rnacompete"),
        grouped=grouped_motifs,
    )


def grouped_motifs(original_spec, group_motifs_path, remove):
    """
    Group motifs together, according to the given file.

    The fire should contain a JSON file containing a list of lists
    of motif names. Each list of motif names will be grouped together

    E.g., [["motif1", "motif2"], ["motif3", "motif4"]] refers
        to grouping together motif1 with motif2 and motif3 with motif4.

    You must provide a grouping for every motif in the original spec, or
        put it in the remove list.

    We just concatenate all the motifs together in each group.
    """
    with open(group_motifs_path) as f:
        grouping = json.load(f)

    remove = set(remove)
    all_in_grouping = set(x for xs in grouping for x in xs)
    assert not (remove & all_in_grouping)
    original = construct(motifs_types(), original_spec)
    back_map = {x: i for i, xs in enumerate(grouping) for x in xs}

    result = defaultdict(list)
    for motif in original:
        if motif in remove:
            continue
        result[back_map[motif]].extend(original[motif])

    result = {f"group_{i:03d}": result[i] for i in result}
    return result
