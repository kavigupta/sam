import os
import urllib
import gzip
from io import StringIO

import numpy as np
import scipy

import tqdm.auto as tqdm
from permacache import permacache, stable_hash

import pandas as pd

from modular_splicing.data_pipeline.spliceai_data_processing_pipeline import (
    create_dataset_from_splice_table,
)
from modular_splicing.utils.sparse_tensor import pad_sparse_motifs_with_cl


@permacache("eclip/download_peaks")
def download_peaks(replicate_category):
    """
    Download all eclip peaks meeting the criteria of belonging to hg19 and K562.
    Onnly using the one cell line for simplicity.

    Returns a dictionary of peak names to the contents of peak files, as text.
    """
    metadata = (
        "https://www.encodeproject.org/metadata/"
        "?control_type%21=%2A&status=released&perturbed=false"
        "&assay_title=eCLIP&assembly=hg19&biosample_ontology.term_name=K562"
        "&type=Experiment&files.assembly=%2A"
    )
    metadata = pd.read_csv(metadata, sep="\t")
    metadata = metadata[
        (metadata["File assembly"] == "hg19")
        & (metadata["File format"] == "bed narrowPeak")
        & (metadata["Biological replicate(s)"] == replicate_category)
    ]
    results = {}
    for _, row in tqdm.tqdm(list(metadata.iterrows())):
        name = row["Experiment target"]
        name, human = name.split("-")
        assert human == "human"
        with urllib.request.urlopen(row["File download URL"]) as f:
            file = gzip.GzipFile(fileobj=f)
            bed_content = file.read().decode("utf-8")
        results[name] = bed_content
    return results


@permacache("eclip/load_peaks")
def load_peaks(*, replicate_category):
    """
    Downloads and converts eclip peak files.

    Returns a dictionary of peak names to dataframes.
    """
    peaks = download_peaks(replicate_category=replicate_category)
    return {motif: load_motif_bed(peaks[motif]) for motif in peaks}


def load_motif_bed(motif_bed_str):
    """
    Uses assumption 1

    Loads the given bed file string as a dataframe
    """
    return pd.read_csv(
        StringIO(motif_bed_str),
        sep="\t",
        header=0,
        names=[
            "chrom",
            "start",
            "end",
            "ident",
            "score",
            "strand",
            "?b",
            "?c",
            "?d",
            "?e",
        ],
    )


@permacache(
    "eclip/classify_eclips",
    key_function=dict(
        genes=lambda x: stable_hash({k: np.array(x[k]).tolist() for k in x}),
        eclips_bed=lambda x: stable_hash({k: np.array(x[k]).tolist() for k in x}),
    ),
)
def classify_eclips(genes, eclips_bed):
    """
    Matches each eclip to a corresponding gene, by chromome, strand, and start/end inclusion.

    Parameters
    ----------
    genes : pandas.DataFrame containing the genes to classify.
        should have columns "chrom", "strand", "start", "end".
    eclips_bed : pandas.DataFrame containing the eclips to classify.
        should have columns "chrom", "strand", "start", "end".

    Returns
    -------
    bad : number of eclips that did not match a gene.
    starts_ends : list for each gene of (start, end) pairs of the eclips that matched it.
    """
    starts_ends = [[] for _ in range(genes.shape[0])]
    bad = 0
    for _, row in tqdm.tqdm(eclips_bed.iterrows(), total=eclips_bed.shape[0]):
        selected = genes[
            (genes.chrom == row.chrom)
            & (genes.strand == row.strand)
            & (genes.start <= row.start)
            & (row.end <= genes.end)
        ]
        if selected.shape[0] == 0:
            bad += 1
        for i in selected.index:
            starts_ends[i].append([row.start, row.end])
    return bad, starts_ends


def attach_eclips_to_genes(genes, eclips_bed):
    """
    Create a new splice table where "donors" and "acceptors" are the beginning and end of each eclip.
    """
    bad, starts_ends = classify_eclips(genes, eclips_bed)
    genes_eclip = genes.copy()
    don_acc = []
    for i in range(genes.shape[0]):
        if starts_ends[i]:
            don_acc.append([",".join(str(t) for t in x) for x in zip(*starts_ends[i])])
        else:
            don_acc.append(["", ""])
    genes_eclip[["donors", "acceptors"]] = don_acc
    return bad, genes_eclip


@permacache(
    "eclip/extract_eclips",
    key_function=dict(
        genes=lambda x: stable_hash({k: np.array(x[k]).tolist() for k in x}),
        eclips_bed=lambda x: stable_hash({k: np.array(x[k]).tolist() for k in x}),
        sequence_path=os.path.abspath,
    ),
)
def extract_eclips(
    genes, eclips_bed, *, sequence_path, data_segment_to_use, data_chunk_to_use
):
    """
    Produce a dataset of eclips from the given genes and eclip bed files.

    Parameters:
        genes: pandas.DataFrame containing the genes to attach the eclips to
        eclips_bed: dictionary from motif name to pandas.DataFrame containing the eclips
        sequence_path: path to the fasta file containing the genome sequence
        data_segment_to_use: which segment of the data to use. Either "train" or "test"
        data_chunk_to_use: whether to use paralogs or not "0" for no, 1 for "yes", "all" for both
    """
    bad, genes_eclip = attach_eclips_to_genes(genes, eclips_bed)

    splice_table_frame = genes_eclip

    CL_max = 10_000
    SL = 5000
    include_seq = False

    def load_file(f):
        out = {}
        for i in range(len(f)):
            key = f"Y{i}"
            if key not in f:
                break
            out[i] = scipy.sparse.csr_matrix(f[key][0].argmax(-1))
        return out

    out = create_dataset_from_splice_table(
        sequence_path=sequence_path,
        splice_table_frame=splice_table_frame,
        load_file=load_file,
        data_segment_to_use=data_segment_to_use,
        data_chunk_to_use=data_chunk_to_use,
        CL_max=CL_max,
        SL=SL,
        include_seq=include_seq,
    )
    return bad, out


def produce_all_eclips(*, replicate_category, is_train, dataset_path, sequence_path):
    """
    Produce a set of eclips for the given splice table and replicate category.

    Parameters:
        replicate_category: which replicate category to use. Either "1" or "2"
        is_train: whether to use the training or test set
        dataset_path: path to the dataset
        sequence_path: path to the genome sequence

    Returns
        total_each: the number of peaks collected per motif
        bad_each: the number of peaks that did not match a gene per motif
        out_each: the gene-aligned eclip dataset per motif. 2 == start of eclip, 1 == end of eclip
    """
    genes = pd.read_csv(
        dataset_path,
        sep="\t",
        names=[
            "gene",
            "chunk",
            "chrom",
            "strand",
            "start",
            "end",
            "donors",
            "acceptors",
        ],
    )
    peaks = load_peaks(replicate_category=replicate_category)
    total_each = {}
    bad_each = {}
    out_each = {}
    for k, bed in tqdm.tqdm(sorted(peaks.items())):
        print(replicate_category, is_train, k)
        bad, out = extract_eclips(
            genes,
            bed,
            sequence_path=sequence_path,
            data_segment_to_use="train" if is_train else "test",
            data_chunk_to_use="all" if is_train else "0",
        )

        total_each[k] = peaks[k].shape[0]
        bad_each[k] = bad
        out_each[k] = out
    return total_each, bad_each, out_each


@permacache(
    "eclip/eclip_pipeline/eclip_dataset_with_spliceai_pipeline",
    key_function=dict(dataset_path=os.path.abspath, sequence_path=os.path.abspath),
)
def eclip_dataset_with_spliceai_pipeline(
    *, replicate_category, is_train, dataset_path, sequence_path, **kwargs
):
    """
    Create the eclip dataset

    Parameters:
        replicate_category: which replicate category to use. Either "1" or "2"
        is_train: whether to use the training or test set
        dataset_path: path to the dataset
        sequence_path: path to the genome sequence

    Returns
        total: the number of peaks collected per motif
        bad: the number of peaks that did not match a gene per motif
        result: a list of data chunks, each of which is a list of sparse matrices
            representing a tensor of shape (N, 2L, M), where M is the number of motifs
            and it is padded out with 0s on either side. Here, 1 == start of motif, 2 == end of motif
    """
    total, bad, ecs = produce_all_eclips(
        replicate_category=replicate_category,
        is_train=is_train,
        dataset_path=dataset_path,
        sequence_path=sequence_path,
        **kwargs,
    )
    motifs_ordered = sorted(ecs)
    result = []
    for dset_idx in range(len(ecs[motifs_ordered[0]])):
        result.append([])
        for motif_idx in range(len(motifs_ordered)):
            a = pad_sparse_motifs_with_cl(
                ecs[motifs_ordered[motif_idx]][dset_idx]
            ).astype(np.int8)
            a.data = 3 - a.data
            result[-1].append(a)
    return total, bad, result
