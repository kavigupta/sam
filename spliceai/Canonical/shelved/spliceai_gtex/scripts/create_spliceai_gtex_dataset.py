import os
import h5py

import numpy as np
import tqdm.auto as tqdm

from shelved.spliceai_gtex.alternative_dataset import (
    data_txt_file_paths,
    COMMON_GENES,
)
from modular_splicing.dataset.alternative.merge_dataset import merged_dataset
from modular_splicing.dataset.alternative.merge_splice_tables import (
    splice_tables_on_common_genes,
)

data_segment_chunks_to_use = [("train", "all"), ("test", "0")]
directory = "../data/gtex_dataset/"

try:
    os.makedirs(directory)
except FileExistsError:
    pass

canonical_data, data_txt_files, _ = splice_tables_on_common_genes(
    data_txt_file_paths, COMMON_GENES
)
x, ys_each = merged_dataset(
    canonical_data=canonical_data,
    data_txt_files={"spliceai_gtex": data_txt_files["spliceai_gtex"]},
    files_for_gene_intersection=["spliceai_gtex"],
    data_segment_chunks_to_use=data_segment_chunks_to_use,
    ref_genome_path="/scratch/kavig/hg19.fa",
)

assert x.keys() == ys_each.keys()

for ch in x:
    with h5py.File(f"{directory}/dataset_{'_'.join(ch)}.h5", "w") as f:
        for key in tqdm.tqdm(x[ch], desc=f"{ch} X"):
            f[key] = np.eye(4, dtype=np.byte)[x[ch][key]]
        for key in tqdm.tqdm(ys_each[ch]["spliceai_gtex"], desc=f"{ch} Y"):
            f[key] = np.eye(3, dtype=np.byte)[
                ys_each[ch]["spliceai_gtex"][key].original
            ][None]
