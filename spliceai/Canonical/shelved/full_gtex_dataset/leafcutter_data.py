import gzip
import os
import numpy as np
import pandas as pd

import tqdm.auto as tqdm
from permacache import permacache, stable_hash
from .fasta import OneHotFasta

from modular_splicing.utils.run_batched import run_batched


@permacache(
    "dataset/gtex/leafcutter_data/load_all_leafcutter_data_2", dict(spm=stable_hash)
)
def load_all_leafcutter_data(spm, dataset_root, fasta_path):
    suffix = "_perind_numers.counts.gz"
    tables = {}
    for fname in tqdm.tqdm(os.listdir(dataset_root)):
        if "numers" not in fname:
            continue
        assert fname.endswith(suffix)
        zip_path = os.path.join(dataset_root, fname)
        table = load_leafcutter(zip_path)
        table = annotate_all_strands(spm, table, genome_fasta_path=fasta_path)
        tables[fname[: -len(suffix)]] = table
    return tables


def load_leafcutter_per_person(zip_path):
    with gzip.GzipFile(zip_path, "r") as f:
        table_numers = pd.read_csv(f, sep=" ")
    assert "chrom" not in table_numers
    new_columns = []
    for col in table_numers.columns:
        gtex, donor_id, *_ = col.split("-")
        assert gtex == "GTEX"
        new_columns.append(donor_id)
    assert len(set(new_columns)) == len(new_columns)
    table_numers.columns = new_columns
    return table_numers


@permacache("dataset/gtex/leafcutter_data/load_leafcutter_1")
def load_leafcutter(zip_path):
    table_numers = load_leafcutter_per_person(zip_path)
    return table_numers.sum(axis=1)


def process_index(x):
    ch, start, end, clu = x.split(":")
    assert clu.startswith("clu_")
    clu = clu[4:]
    return dict(chr=ch, start=int(start) - 1, end=int(end) - 1, cluster=int(clu))


def separate_columns(table):
    return pd.DataFrame(
        [{**process_index(x), "count": el} for x, el in zip(table.index, table)]
    )


@permacache(
    "dataset/gtex/leafcutter_data/annotate_all_strands_3",
    key_function=dict(table=stable_hash, spm=stable_hash),
)
def annotate_all_strands(spm, table, *, genome_fasta_path, cl=50):
    genome_one_hot_path = genome_fasta_path.replace(".fa", "_one_hot.pkl")
    genome = OneHotFasta(genome_fasta_path, genome_one_hot_path)
    table = separate_columns(table).copy()
    table["strand"] = table["acc"] = table["don"] = table["acc_other"] = table[
        "don_other"
    ] = np.nan
    for k in tqdm.tqdm(sorted(set(table.chr))):
        annotate_strand(genome, spm, table, chrom=k, cl=cl)
    return table


def annotate_strand(genome, spm, table, *, chrom, cl):
    table_for_chr = table[table.chr == chrom]
    yps_starts, yps_ends = [
        [
            run_batched(
                spm.forward_just_splicepoints,
                genome.load(
                    chrom,
                    np.array(se),
                    cl,
                    reverse_complement=rc,
                ).astype(np.float32),
                1000,
            )
            for rc in (False, True)
        ]
        for se in (table_for_chr.start, table_for_chr.end)
    ]
    acc_minus = yps_starts[1][:, cl // 2, 0]
    don_plus = yps_starts[0][:, cl // 2, 1]

    acc_plus = yps_ends[0][:, cl // 2, 0]
    don_minus = yps_ends[1][:, cl // 2, 1]

    score_minus = acc_minus + don_minus
    score_plus = acc_plus + don_plus

    score_table = pd.DataFrame(
        dict(cluster=table_for_chr.cluster, rel_minus=score_minus - score_plus)
    )

    is_minus_by_cluster = score_table.groupby("cluster").mean() > 0

    is_minus_by_cluster = dict(
        zip(is_minus_by_cluster.index, is_minus_by_cluster.rel_minus)
    )

    is_minus = np.array([is_minus_by_cluster[c] for c in table_for_chr.cluster])

    strands = np.where(is_minus, "-", "+")
    acc = np.where(is_minus, acc_minus, acc_plus)
    acc_other = np.where(is_minus, acc_plus, acc_minus)
    don = np.where(is_minus, don_minus, don_plus)
    don_other = np.where(is_minus, don_plus, don_minus)
    table.loc[table.chr == chrom, "strand"] = strands
    table.loc[table.chr == chrom, "acc"] = acc
    table.loc[table.chr == chrom, "don"] = don
    table.loc[table.chr == chrom, "acc_other"] = acc_other
    table.loc[table.chr == chrom, "don_other"] = don_other
