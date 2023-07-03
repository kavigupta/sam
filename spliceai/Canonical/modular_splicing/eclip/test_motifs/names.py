from types import SimpleNamespace

from ..data.pipeline import load_peaks
from modular_splicing.motif_names import get_motif_names


def peak_names(*, replicate_category):
    """
    Get the names of the peaks for a given replicate category.
    """
    return sorted(load_peaks(replicate_category=replicate_category))


def get_testing_names(*, motif_names_source):
    """
    Return a namespace containing the following attributes:

    - common_names: the names of the motifs that are common to both the motif name source and eclip
    - eclip_names: the names of all eclip motifs
    - motif_names: the names of all motifs in the motif name source
    - eclip_idxs: the indexes of all the common motifs in the eclip names order
    - motif_idxs: the indexes of all the common motifs in the motif names source order
    """
    eclip_names = peak_names(replicate_category="1")
    motif_names = get_motif_names(motif_names_source)
    common_names = sorted(set(eclip_names) & set(motif_names))
    eclip_idxs = [eclip_names.index(motif) for motif in common_names]
    motif_idxs = [motif_names.index(motif) for motif in common_names]
    return SimpleNamespace(
        common_names=common_names,
        eclip_names=eclip_names,
        motif_names=motif_names,
        eclip_idxs=eclip_idxs,
        motif_idxs=motif_idxs,
    )
