import functools
import os

from .data_directory import DATA_DIRECTORY
from .rbns import read_rbns_motif_from_file, read_rbns_motifs

RBNS_V2_EXTRAS_DIRECTORY = os.path.join(DATA_DIRECTORY, "motifs_rbns_v2_extras")
PASSED_QC = (
    "CIRBP ELAVL1 ENOX1 ESRP1 EXOSC2 EXOSC4 EXOSC8 IGF2BP3 LIN28B NSUN2"
    " PABPC3 PPP1R10 RBM11 RBM20 RBM24 RBM3 RBM5 RBMY1A1 RPS5 SF3B6 SNRPB2 SRRM2"
    " SUCLG1 SYNCRIP TDRD10 THUMPD1 TIAL1 TIS11B TIS11D TRA2B TROVE2 XRCC6 XRN2"
    " YBX1 YBX3 ZC3H11A ZC3H15 ZC3H18"
).split(" ")


@functools.lru_cache(None)
def read_rbns_motifs_v2_extras_without_passing_qc(directory):
    """
    Read all the RBNS v2 extras motifs from the given directory.
    """
    return {
        protein_path.split(".")[0]: read_rbns_motif_from_file(
            os.path.join(directory, protein_path),
            has_runinfo=False,
        )
        for protein_path in os.listdir(directory)
    }


def read_rbns_motifs_v2_extras(directory=RBNS_V2_EXTRAS_DIRECTORY):
    """
    Read the RBNS v2 extras motifs from the given directory that passed QC.
    """
    motifs = read_rbns_motifs_v2_extras_without_passing_qc(directory)
    return {name: motifs[name] for name in PASSED_QC}


def read_rbns_v2p1_motifs():
    """
    Read the RBNS v2.1 motifs.

    These motifs consist of the RBNS motifs but with the RBNS v2 extras motifs added in.

    We only include the ones that passed QC. There is some overlap between these extra
    ones and the RBNS ones, so we default to using the RBNS ones.
    """
    motifs = {}
    motifs.update(read_rbns_motifs())
    # remove overlaps
    del motifs["ESRP1"]
    del motifs["RBM24"]
    extras = read_rbns_motifs_v2_extras()
    assert set(extras) & set(motifs) == set()
    motifs.update(extras)
    return motifs
