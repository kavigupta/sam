from collections import defaultdict
import os
import re
import functools

import numpy as np

from modular_splicing.psams.sources.rbns import read_rbns_motifs

from ..psams import PSAM
from .data_directory import DATA_DIRECTORY

RNA_COMPETE_MOTIFS_DIRECTORY = os.path.join(DATA_DIRECTORY, "motifs_rnacompete")


def rna_compete_motifs_for(species_list):
    """
    Produce a dictionary of RNAcompete motifs for the given specieses
    """
    motifs = read_raw_rnacompete_motifs()
    motifs_selected = {}
    for full_name in motifs:
        name, sp = parse_rnacompete_name(full_name)
        if sp in species_list:
            motifs_selected[name] = motifs[full_name]
    return motifs_selected


def rbns_rnacompete_motifs_deduplicated(overlap_mode="merge"):
    """
    Return a dictionary of RNAcompete motifs, cleaned up (invertebrates removed),
        and deduplicated with RBNS (if overlap_mode is "merge", or "use-rnacompete" or "use-rbns").

    Possible values for overlap_mode:
        "merge": merge the RBNS and RNAcompete motifs when they overlap, into a single motif.
        "use-rnacompete": use the RNAcompete motifs when they overlap with RBNS motifs.
        "use-rbns": use the RBNS motifs when they overlap with RNAcompete motifs.
        "just-rnacompete": use only the RNAcompete motifs.
    """
    vertebrates = {
        "Mus_musculus": "mm",
        "Gallus_gallus": "gg",
        "Xenopus_tropicalis": "xt",
        "Danio_rerio": "dr",
        "Tetraodon_nigroviridis": "tn",
    }
    invertebrates = {
        "Trypanosoma_brucei",
        "Drosophila_melanogaster",
        "Trichomonas_vaginalis",
        "Plasmodium_falciparum",
        "Leishmania_major",
        "Saccharomyces_cerevisiae",
        "Caenorhabditis_elegans",
        "Aspergillus_nidulans",
        "Phytophthora_ramorum",
        "Physcomitrella_patens",
        "Schistosoma_mansoni",
        "Rhizopus_oryzae",
        "Neurospora_crassa",
        "Arabidopsis_thaliana",
        "Thalassiosira_pseudonana",
        "Nematostella_vectensis",
        "Ostreococcus_tauri",
        "Naegleria_gruberi",
    }
    motifs = defaultdict(list)
    if overlap_mode != "just-rnacompete":
        motifs.update({k: list(v) for k, v in read_rbns_motifs().items()})
    for fullname, mot in read_raw_rnacompete_motifs().items():
        name, species = parse_rnacompete_name(fullname)
        if species == "Homo_sapiens":
            if overlap_mode == "merge":
                motifs[name].extend(mot)
            elif overlap_mode == "use-rbns":
                if not motifs[name]:
                    motifs[name] = mot
            elif overlap_mode == "use-rnacompete":
                motifs[name] = mot
            elif overlap_mode == "just-rnacompete":
                assert not motifs[name]
                motifs[name] = mot
            else:
                raise RuntimeError(f"Invalid overlap mode: {overlap_mode}")
            continue
        if species in invertebrates:
            continue
        if species in vertebrates:
            name = name + "_" + vertebrates[species]
            assert name not in motifs
            motifs[name] = mot
            continue
        raise RuntimeError("Unrecognized species")
    return motifs


def read_raw_rnacompete_motifs(path=RNA_COMPETE_MOTIFS_DIRECTORY):
    """
    Read raw RNAcompete motifs from the given directory.
    """
    files = os.listdir(path)
    by_prefix = defaultdict(lambda: defaultdict(lambda: None))
    for f in files:
        if f == "README.txt":
            continue
        *prefix, numeral = os.path.splitext(f)[0].split("_")
        try:
            numeral = int(numeral)
            prefix = "_".join(prefix)
        except:
            prefix = "_".join(prefix + [numeral])
            numeral = 1
        numeral -= 1
        by_prefix[prefix][numeral] = os.path.join(RNA_COMPETE_MOTIFS_DIRECTORY, f)
    result = {}
    for prefix, by_indices in by_prefix.items():
        max_n = max(
            read_motif_rna_compete(by_indices[i]).n for i in range(len(by_indices))
        )
        result[prefix] = [
            read_motif_rna_compete(by_indices[i], max_n) for i in range(len(by_indices))
        ]
    return result


@functools.lru_cache(None)
def read_motif_rna_compete(path, min_length=0):
    """
    Read an RNAcompete motif from the given file.
    """
    with open(path) as f:
        lines = [l.split() for l in f]
    assert lines[0] == ["Pos", "A", "C", "G", "U"]
    matr = []
    for i, lin in enumerate(lines[1:]):
        assert i + 1 == int(lin[0])
        matr.append([float(x) for x in lin[1:]])
    matr = np.array(matr)
    matr = matr / matr.max(1)[:, None]
    if min_length > matr.shape[0]:
        pad = min_length - matr.shape[0]
        left_pad = pad // 2
        right_pad = pad - left_pad
        matr = np.concatenate(
            [np.ones((left_pad, 4)), matr, np.ones((right_pad, 4))], axis=0
        )
    return PSAM(1, matr.shape[0], None, None, None, matr)


def parse_rnacompete_name(fullname):
    mat = re.match("^(.*)_((.*)_(.*))$", fullname)
    name, species = mat.group(1), mat.group(2)
    return name, species
