from functools import lru_cache
import os

import pandas as pd

from .data_directory import DATA_DIRECTORY
from .rbns import read_rbns_motifs
from .rbns_v2 import read_rbns_v2p1_motifs

RBNS_MOTIF_FUNCTIONS = os.path.join(DATA_DIRECTORY, "motif_functions")
RBNS_MOTIF_FUNCTIONS_v1p0 = os.path.join(RBNS_MOTIF_FUNCTIONS, "rbns_1p0_functions.csv")
RBNS_MOTIF_FUNCTIONS_v2p1 = os.path.join(RBNS_MOTIF_FUNCTIONS, "rbns_2p1_functions.csv")


def read_rbns_motif_functions_v1p0():
    """
    Read the RBNS motif functions for v1.0. These were sent by Chris
        a while back.
    """
    v1po_rename = {
        "Gene": "RBP",
        "Splicing regulation": "splicing_regulation",
        "Spliceosome": "spliceosome",
    }
    table = pd.read_csv(RBNS_MOTIF_FUNCTIONS_v1p0)
    table = table[[x for x in v1po_rename.keys()]]
    table = table.rename(columns=v1po_rename)

    return table


def read_rbns_motif_functions_v2p1():
    """
    Read the RBNS motif functions for v2.1. These were sent by Jennifer
        in October 2022.
    """
    table = pd.read_csv(RBNS_MOTIF_FUNCTIONS_v2p1)
    assert (
        table.columns
        == [
            "Unnamed: 0",
            "Regulation of Splicing: GO: 0043484",
            "spliceosome: GO:0005681",
        ]
    ).all()
    table.columns = ["RBP", "splicing_regulation", "spliceosome"]
    table.RBP = table.RBP.apply(lambda x: "PPP1R10" if x == "PPR1R10" else x)
    return table


@lru_cache(None)
def read_rbns_motif_functions():
    """
    Consolidate the RBNS motif functions for v1.0 and v2.1. Check that
        the RBPs are the same.
    """
    one, two = read_rbns_motif_functions_v1p0(), read_rbns_motif_functions_v2p1()
    for table in [one, two]:
        assert len(table.RBP) == len(set(table.RBP))
    one, two = one.set_index("RBP"), two.set_index("RBP")
    overlaps = set(one.index).intersection(set(two.index))
    for overlap in overlaps:
        assert (one.loc[overlap] == two.loc[overlap]).all()
    one = one[[x for x in one.columns if x not in overlaps]]
    combination = pd.concat([one, two], axis=0)
    return combination


@lru_cache(None)
def is_functional_in_splicing():
    """
    Produce a dictionary mapping RBPs to whether they are functional in
        splicing.
    """
    table = read_rbns_motif_functions()
    table = table.T.any()
    return dict(zip(table.index, table.values))


def filter_for_functionality(motifs):
    """
    Filter the RBNS motifs to only those that are functional in splicing.
    """
    functional = is_functional_in_splicing()
    return {
        name: motif
        for name, motif in motifs.items()
        if name in {"3P", "5P"} or functional[name]
    }


def read_rbns_functional_motifs(**kwargs):
    """
    Read the RBNS motifs that are functional in splicing.
    """
    return filter_for_functionality(read_rbns_motifs(**kwargs))


def read_rbns_v2p1_functional_motifs(**kwargs):
    """
    Read the RBNS 2.1 motifs that are functional in splicing.
    """
    return filter_for_functionality(read_rbns_v2p1_motifs(**kwargs))
