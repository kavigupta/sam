import json
import os
import pickle
import tempfile

import pandas as pd
from .leafcutter_data import load_all_leafcutter_data

from modular_splicing.data_pipeline.create_datafile import produce_datafiles
from modular_splicing.data_pipeline.create_dataset import produce_datasets_from_datafile

from .compute_psi import (
    COORDINATE_COLUMNS,
    all_psi_values,
    annotate_chunks,
    annotate_splice_table_with_indices,
)


def produce_splice_table(all_keys, splice_table_path):
    coordinate_table = pd.DataFrame(all_keys, columns=COORDINATE_COLUMNS)
    splice_table, matched, counts_each = annotate_splice_table_with_indices(
        splice_table_path, coordinate_table
    )
    splice_table = splice_table.copy()
    splice_table.insert(
        1,
        "chunk",
        annotate_chunks(splice_table[["name", "chr", "strand", "start", "end"]]),
    )

    return coordinate_table, splice_table, matched, counts_each


def save_psi_indexed(
    spm, *, dataset_root, fasta_path, splice_table_path, ws, data_path_folder
):
    tissue_names = f"{data_path_folder}/tissue_names.json"
    path_psi = f"{data_path_folder}/psis.pkl"
    path_splice_table = f"{data_path_folder}/splice_table.csv"
    if all(os.path.exists(p) for p in [tissue_names, path_psi, path_splice_table]):
        print("PSI already exists, skipping creation.")
        return path_splice_table
    tables = load_all_leafcutter_data(spm, dataset_root, fasta_path)
    with open(tissue_names, "w") as f:
        json.dump(list(tables.keys()), f)
    all_keys, psi_values = all_psi_values(spm, dataset_root, fasta_path, ws=ws)
    _, splice_table, _, _ = produce_splice_table(all_keys, splice_table_path)
    with open(path_psi, "wb") as f:
        pickle.dump(dict(all_keys=all_keys, psi_values=psi_values), f)

    splice_table.to_csv(path_splice_table, index=False)
    return path_splice_table


def produce_datasets(
    spm,
    *,
    leafcutter_dataset_root,
    fasta_path,
    splice_table_path,
    data_path_folder,
    CL_max,
    segment_chunks,
    ws,
):
    splice_table_path = save_psi_indexed(
        spm,
        dataset_root=leafcutter_dataset_root,
        fasta_path=fasta_path,
        splice_table_path=splice_table_path,
        data_path_folder=data_path_folder,
        ws=ws,
    )
    produce_datafiles(
        splice_table=pd.read_csv(splice_table_path),
        fasta_path=fasta_path,
        data_path_folder=data_path_folder,
        CL_max=CL_max,
        segment_chunks=segment_chunks,
    )
    produce_datasets_from_datafile(
        data_path_folder=data_path_folder,
        segment_chunks=segment_chunks,
        CL_max=CL_max,
        SL=5000,
    )
