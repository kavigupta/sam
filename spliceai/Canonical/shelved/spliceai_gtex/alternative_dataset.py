import pandas as pd
import h5py

import tqdm.auto as tqdm
import numpy as np
from modular_splicing.dataset.alternative.merge_splice_tables import (
    splice_tables_on_common_genes,
)

from modular_splicing.dataset.alternative.merge_dataset import merged_dataset


data_txt_file_paths = {
    "spliceai_canonical": "canonical_dataset.txt",
    "spliceai_gtex": "../data/gtex_dataset.txt",
    "merkin_evo_new": "../data/evo_alt_exons/evolutionarily_new_exons_hg19_2015_Merkin_et_al.txt",
    "merkin_evo_old": "../data/evo_alt_exons/evolutionarily_old_exons_hg19_2015_Merkin_et_al.txt",
    "merkin_alt": "../data/evo_alt_exons/alternative_exons_hg19_2015_Merkin_et_al.txt",
    "merkin_con": "../data/evo_alt_exons/constitutive_exons_hg19_2015_Merkin_et_al.txt",
    "mazin_evo_new": "../data/evo_alt_exons/evolutionarily_new_exons_hg19_2021_Mazin_et_al.txt",
}

COMMON_GENES = "spliceai_canonical"


def split_out_exons(data_txt_files):
    exons_by_gene = {
        k: {r.Name: set(match_exons(r)) for _, r in data_txt_files[k].iterrows()}
        for k in data_txt_files
    }
    exons_flat = {
        k: {
            (name, which, coord)
            for name, exons in exons_by_gene[k].items()
            for which, coord in exons
        }
        for k in data_txt_files
    }
    return exons_flat


def match_exons(r):
    s, e = r.sstarts.split(","), r.sends.split(",")
    last_s, last_e = s.pop(), e.pop()
    assert last_s == "" and last_e == ""
    for x in s:
        yield "start", x

    for x in e:
        yield "end", x


def auto_intersections(genes_each, *, divide_cols=False):
    return pd.DataFrame(
        {
            k1: {
                k2: len(genes_each[k1] & genes_each[k2])
                / (len(genes_each[k1]) if divide_cols else 1)
                for k2 in genes_each
            }
            for k1 in genes_each
        }
    )


def write_alternative_dataset(
    *,
    files_for_gene_intersection,
    data_segment_chunks_to_use,
    ref_genome_path,
    out_prefix,
    include_splice_tables_from=None,
):
    """
    Creates a dataset with the alternative splice sites
        Will create a dataset for each element of data_segment_chunks_to_use
        Genes will be selected based on which appear in all of files_for_gene_intersection
            and will be placed in the order of spliceai_canonical.
        The X values will just be as normal, while the Y values will be
            one hot encoded, in the order given by the key "ordering".
            The bottom channel will be left off, so it will be

        spliceai_canonical's acceptor
        spliceai_canonical's donor
        spliceai_gtex's acceptor
        spliceai_gtex's donor
    """
    canonical_data, data_txt_files, _ = splice_tables_on_common_genes(
        data_txt_file_paths, COMMON_GENES
    )

    if include_splice_tables_from is not None:
        data_txt_files = {
            k: data_txt_files[k]
            for k in data_txt_files
            if k in include_splice_tables_from
        }

    x, ys_each = merged_dataset(
        canonical_data=canonical_data,
        data_txt_files=data_txt_files,
        files_for_gene_intersection=files_for_gene_intersection,
        data_segment_chunks_to_use=data_segment_chunks_to_use,
        ref_genome_path=ref_genome_path,
    )
    for ch in ys_each:
        ordering = list(ys_each[ch].keys())
        name = f"{out_prefix}_{ch[0]}_{ch[1]}"
        with h5py.File(f"{name}.h5", "w") as out:
            out.create_dataset("ordering", data=np.array(ordering, dtype="S"))
            ye_ch = ys_each[ch]
            keys = [ye_ch[k].keys() for k in ye_ch]
            for key in keys:
                assert key == keys[0]
            keys = keys[0]
            for key in tqdm.tqdm(keys, desc=str(ch)):
                out[key] = np.concatenate(
                    [
                        np.eye(3, dtype=np.byte)[ye_ch[k][key].original][:, :, 1:]
                        for k in ordering
                    ],
                    axis=-1,
                )
            for key in tqdm.tqdm(x[ch], desc=str(ch)):
                out[key] = np.eye(4, dtype=np.byte)[x[ch][key]]
