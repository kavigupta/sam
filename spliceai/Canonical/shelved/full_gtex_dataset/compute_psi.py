import numpy as np
import pandas as pd

from permacache import permacache, stable_hash, drop_if_equal

import tqdm.auto as tqdm
from .leafcutter_data import load_all_leafcutter_data

from modular_splicing.utils.construct import construct

COORDINATE_COLUMNS = ["chrom", "strand", "pos", "is_end"]


def convert_psi_table(t, ws):
    keys = list(zip(*(t[col] for col in COORDINATE_COLUMNS)))
    values = [dict(zip(ws, vs)) for vs in zip(*(t[f"psi_estimate_{w}"] for w in ws))]
    return dict(zip(keys, values))


def filter_cluster_sizes(table, min_cluster_size, max_cluster_size):
    cluster_size = table["cluster"].map(table["cluster"].value_counts())
    return table[
        (min_cluster_size <= cluster_size) & (cluster_size <= max_cluster_size)
    ]


@permacache(
    "dataset/gtex/compute_psi/all_psi_values_5",
    dict(
        spm=stable_hash,
        min_cluster_size=drop_if_equal(0),
        max_cluster_size=drop_if_equal(float("inf")),
    ),
)
def all_psi_values(
    spm,
    dataset_root,
    fasta_path,
    *,
    ws,
    min_cluster_size=0,
    max_cluster_size=float("inf"),
):
    tables = load_all_leafcutter_data(spm, dataset_root, fasta_path)
    tables = {
        k: filter_cluster_sizes(tables[k], min_cluster_size, max_cluster_size)
        for k in tables
    }
    psi_tables = {k: compute_psi(tables[k], ws=ws) for k in tqdm.tqdm(tables)}
    processed = {
        k: convert_psi_table(psi_tables[k], ws=ws) for k in tqdm.tqdm(psi_tables)
    }
    all_keys = set()
    for p in processed.values():
        all_keys.update(p)
    all_keys = sorted(all_keys)
    psi_values = [
        {k: processed[k][key] for k in processed if key in processed[k]}
        for key in tqdm.tqdm(all_keys)
    ]
    return all_keys, psi_values


@permacache(
    "dataset/gtex/compute_psi/compute_psi", key_function=dict(table=stable_hash)
)
def compute_psi(table, ws, method=dict(type="using_local_max_coverage")):
    rows = []
    for chrom in tqdm.tqdm(sorted(set(table.chr))):
        rows += construct(
            dict(
                using_local_max_coverage=compute_psi_for_chrom_using_local_max_coverage
            ),
            method,
            chrom=chrom,
            table_for_chrom=table[table.chr == chrom],
            ws=ws,
        )
    return pd.DataFrame(rows)


def compute_psi_for_chrom_using_local_max_coverage(chrom, table_for_chrom, ws):
    size = 1 + table_for_chrom[["start", "end"]].max().max()
    # strand * is_end * pos -> count
    introns_covering = np.zeros((2, size), dtype=np.int)
    introns_ending = np.zeros((2, 2, size), dtype=np.int)
    for _, row in table_for_chrom.iterrows():
        strand_id = {"-": 0, "+": 1}[row.strand]
        introns_covering[strand_id, row.start : row.end + 1] += row["count"]
        introns_ending[strand_id, [0, 1], [row.start, row.end]] += row["count"]
    rows = []
    for strand_id, is_end, pos in np.array(np.where(introns_ending)).T:
        row = dict(
            chrom=chrom, strand={0: "-", 1: "+"}[strand_id], pos=pos, is_end=is_end
        )
        row.update(
            {
                f"psi_estimate_{w}": introns_ending[strand_id, is_end, pos]
                / introns_covering[strand_id, max(pos - w, 0) : pos + w + 1].max()
                for w in ws
            }
        )
        rows.append(row)
    return rows


@permacache(
    "dataset/gtex/compute_psi/annotate_splice_table_with_indices_4",
    key_function=dict(coordinate_table=stable_hash),
)
def annotate_splice_table_with_indices(splice_table_path, coordinate_table):
    # highly unclear why this is necessary, but without it the data just doesn't line up right
    # with the genome. Possible there's an off-by-one error somewhere else in the spliceai processing
    # code? Or perhaps something is one-indexed here. In any case, with this change, the splicepoints
    # line up appropriately. See notebook `notebooks/gtex-dataset/check-sequence-logos.ipynb`
    # for confirmation
    coordinate_table = coordinate_table.copy()
    coordinate_table.pos += 1

    splice_table = pd.read_csv(splice_table_path)
    splice_table = splice_table[splice_table.chr != "chrM"].copy()

    assert set(splice_table.chr) == set(coordinate_table.chrom)
    filt_tables = {
        (chrom, strand): coordinate_table[
            (coordinate_table.chrom == chrom) & (coordinate_table.strand == strand)
        ]
        for chrom in set(coordinate_table.chrom)
        for strand in set(coordinate_table.strand)
    }
    matched = set()
    counts_each = []
    extras = []
    for _, row in tqdm.tqdm(list(splice_table.iterrows())):
        filt_table = filt_tables[row.chr, row.strand]
        filt_table = filt_table[
            (filt_table.pos <= row.end) & (filt_table.pos >= row.start)
        ]
        starts_ends = (
            filt_table[filt_table.is_end == 0],
            filt_table[filt_table.is_end == 1],
        )
        positions = [",".join(str(x) for x in t.pos) for t in starts_ends]

        identifiers = [",".join(str(x) for x in t.index) for t in starts_ends]

        matched |= set(filt_table.index)
        counts_each.append([x.shape[0] for x in starts_ends])

        extras.append(positions + identifiers)
    for name, vals in zip(["start_pos", "end_pos", "start_id", "end_id"], zip(*extras)):
        splice_table[name] = vals
    return splice_table, matched, counts_each


@permacache(
    "dataset/gtex/compute_psi/annotate_chunks_1",
    key_function=dict(coordinate_table=stable_hash),
)
def annotate_chunks(coordinate_table):
    summary = ""
    for k in coordinate_table:
        summary = summary + "," + coordinate_table[k].apply(str)
    chunk = summary.apply(lambda x: int(stable_hash(x), 16) % 2)
    return chunk
