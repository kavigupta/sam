from functools import lru_cache
import io
from types import SimpleNamespace

import numpy as np
import scipy.io as sio

from permacache import permacache

from modular_splicing.utils.download import read_gzip


def read_mrs(name):
    """
    Read the MRS data from my public fork of the paper's data.
    """
    prefix = "https://github.com/kavigupta/cell-2015/raw/ca54d1117fd28375260bfde3d1b46f3d6074f306/"
    return read_gzip(f"{prefix}/data_gz/{name}.gz")


@permacache("modular_splicing/mrs/load/read_mrs_sequences")
def read_mrs_sequences(name):
    """
    Read the MRS sequences from the CSV file from the paper's data.

    Arguments:
        name: The name of the file to read, e.g. "A3SS_seqs.csv"

    Returns:
        tags: The tags for each sequence.
        seqs: The sequences.
    """
    x = read_mrs(name)
    x = (
        x.replace(b"A", b"0")
        .replace(b"C", b"1")
        .replace(b"G", b"2")
        .replace(b"T", b"3")
    )
    x = x.split(b"\n")[1:-1]
    x = [t.split(b",") for t in x]
    x = [[list(map(int, e)) for e in t] for t in x]
    tags, seqs = zip(*x)
    tags, seqs = np.array(tags, dtype=np.int8), np.array(seqs, dtype=np.int8)
    tags, seqs = tags - ord("0"), seqs - ord("0")
    return tags, seqs


@permacache("modular_splicing/mrs/load/gather_data_2")
def gather_data():
    """
    Gather the data from the paper.

    Returns:
        result, which has keys 3 and 5, and values that are SimpleNamespace
        objects containing reads, sequences, and tags.
    """
    data = sio.loadmat(io.BytesIO(read_mrs("Reads.mat")))

    result = {}
    for k in 3, 5:
        tags, seqs = read_mrs_sequences(f"A{k}SS_Seqs.csv")
        result[k] = SimpleNamespace(tags=tags, seqs=seqs, reads=data[f"A{k}SS"])
    return result


def isoforms():
    """
    Produces isoforms for both the donor and acceptor.

    Copied from
    https://github.com/kavigupta/cell-2015/blob/master/ipython.notebooks/Cell2015_N1_Library_Statistics.ipynb

    """
    A5SS_data, A3SS_data = gather_data()[5].reads, gather_data()[3].reads

    A5SS_reads_per_plasmid = np.array(A5SS_data.sum(1)).flatten()
    A3SS_reads_per_plasmid = np.array(A3SS_data.sum(1)).flatten()

    A5SS_fraction = np.array(A5SS_data.todense())
    A5SS_fraction = (A5SS_fraction.T / A5SS_fraction.sum(1)).T
    # Remove plasmids with no reads
    A5SS_fraction = A5SS_fraction[A5SS_reads_per_plasmid > 0]

    A5_isoforms = {}
    A5_isoforms["SD_1"] = A5SS_fraction[:, 0]
    A5_isoforms["SD_2"] = A5SS_fraction[:, 44]
    A5_isoforms["SD_CRYPT"] = A5SS_fraction[:, 79]
    A5_isoforms["SD_NEW"] = A5SS_fraction[:, 7:35].sum(axis=1) + A5SS_fraction[
        :, 50:75
    ].sum(axis=1)
    A5_isoforms["No_SD"] = A5SS_fraction[:, 303]

    nn = A3SS_reads_per_plasmid > 0

    A3_isoforms = {}
    A3_isoforms["SA_1"] = (
        np.array(A3SS_data[:, 235].todense()).reshape(-1).astype(np.float64)[nn]
        / A3SS_reads_per_plasmid[nn]
    )
    A3_isoforms["SA_2"] = (
        np.array(A3SS_data[:, 388].todense()).reshape(-1).astype(np.float64)[nn]
        / A3SS_reads_per_plasmid[nn]
    )
    A3_isoforms["SA_CRYPT"] = (
        np.array(A3SS_data[:, 388 - 16].todense()).reshape(-1).astype(np.float64)[nn]
        / A3SS_reads_per_plasmid[nn]
    )
    A3_isoforms["SA_NEW"] = (
        np.array(
            A3SS_data[:, 388 - 19 - 25 : 388 - 19].sum(axis=1)
            + A3SS_data[:, 388 + 3 : 388 + 28].sum(axis=1)
        )
        .reshape(-1)
        .astype(np.float64)[nn]
        / A3SS_reads_per_plasmid[nn]
    )
    A3_isoforms["No_SA"] = (
        np.array(A3SS_data[:, 0].todense()).reshape(-1).astype(np.float64)[nn]
        / A3SS_reads_per_plasmid[nn]
    )

    return {
        5: (A5SS_reads_per_plasmid > 0, A5_isoforms),
        3: (A3SS_reads_per_plasmid > 0, A3_isoforms),
    }


def relative_intron_enrichment():
    """
    Produce a mask and relative intron enrichment for both the donor and acceptor.
    """
    iso = isoforms()
    mask_5, A5_isoforms = iso[5]
    mask_3, A3_isoforms = iso[3]

    return {
        5: (mask_5, A5_isoforms["SD_1"] - A5_isoforms["SD_2"]),
        3: (mask_3, A3_isoforms["SA_2"] - A3_isoforms["SA_1"]),
    }


def donor_sequences(z, mask):
    """
    Produce the donor sequences for the given mask.
    """
    don_seqs = z[5].seqs[mask]
    don_seqs = don_seqs[:, : 7 + 25 + 10]
    return don_seqs


def acceptor_sequences(z, mask):
    """
    Produce the acceptor sequences for the given mask.
    """
    # from here https://github.com/kavigupta/cell-2015/blob/master/ipython.notebooks/Cell2015_N0A_A3SS_Fastq_to_Spliced_Reads.ipynb
    # we can see that the second degenerate region is followed by
    # NNNNNNNNNNNNNNNNNNNNNNNNN atgatt acac
    # the first 6 bases are present, and we attach ACAC to the end
    acc_seqs = z[3].seqs[mask]
    acc_seqs = acc_seqs[:, 39:]
    acc_seqs = np.concatenate(
        [acc_seqs, np.repeat([[0, 1, 0, 1]], acc_seqs.shape[0], axis=0)], axis=1
    )
    return acc_seqs


@lru_cache(None)
def mrs_data():
    """
    Load the MRS data.
    """
    z = gather_data()
    rie = relative_intron_enrichment()
    mask_5, rel_enrich_5 = rie[5]
    mask_3, rel_enrich_3 = rie[3]

    don_seqs = donor_sequences(z, mask_5)
    acc_seqs = acceptor_sequences(z, mask_3)

    return {
        5: dict(x=don_seqs, y=rel_enrich_5),
        3: dict(x=acc_seqs, y=rel_enrich_3),
    }
