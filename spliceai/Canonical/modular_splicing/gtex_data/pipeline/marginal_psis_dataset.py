import json
import pickle

import numpy as np
import pandas as pd
from modular_splicing.utils.construct import construct

from shelved.full_gtex_dataset.pipeline import (
    produce_datasets_from_datafile,
)

from modular_splicing.data_pipeline.create_datafile import produce_datafiles
from modular_splicing.gtex_data.annotation.compute_psis_from_annotation import (
    compute_psis_all,
)

from .marginal_psis import compute_psis_from_junction_data
from .splice_table import produce_splice_table
from .gtex_data_table import transcript_data


def generate_probabilistic_gtex_dataset(
    data_path_folder, fasta_path, CL_max, SL, segment_chunks, **kwargs
):
    """
    Generate the probabilistic gtex dataset.

    Parameters
    ----------
    data_path_folder : str
        The data path folder.
    fasta_path : str
        The fasta path.
    CL_max : int
        The maximum chunk length.
    SL : int
        The segment length.
    segment_chunks : int
        The number of segment chunks.
    **kwargs
        The keyword arguments to pass into output_psis.
    """
    splice_table_path, bad_genes = output_psis(
        data_path_folder=data_path_folder,
        CL_max=CL_max,
        **kwargs,
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
        SL=SL,
    )
    return bad_genes


def output_psis(
    data_path_folder,
    *,
    psi_computation_spec,
    ensembl_version,
    tissue_id_function,
    sample=None,
    CL_max,
):
    """
    Output the psi values.

    Parameters
    ----------
    data_path_folder : str
        The data path folder.
    psi_computation_spec: dict
        The psi computation spec.
    ensembl_version : int
        The ensembl version.
    tissue_id_function : function
        The tissue id function, which maps the tissue name to the key.
    sample : int, optional
        The sample size to use for testing. If None, use the full dataset.
    CL_max : int
        The CL max.
    """
    path_tissue_names = f"{data_path_folder}/tissue_names.json"
    path_psi = f"{data_path_folder}/psis.pkl"
    path_splice_table = f"{data_path_folder}/splice_table.csv"
    splice_table, tissue_names, psi_dump, bad_genes = annotate_splice_table(
        ensembl_version=ensembl_version,
        tissue_id_function=tissue_id_function,
        CL_max=CL_max,
        psi_computation_spec=psi_computation_spec,
    )

    with open(path_tissue_names, "w") as f:
        json.dump(list(tissue_names), f)

    if sample is not None:
        splice_table = splice_table.iloc[
            np.random.RandomState(0).choice(splice_table.shape[0], size=sample)
        ]
    with open(path_psi, "wb") as f:
        pickle.dump(psi_dump, f)

    splice_table.to_csv(path_splice_table, index=False)
    return path_splice_table, bad_genes


def annotate_splice_table(
    *, psi_computation_spec, ensembl_version, tissue_id_function, CL_max
):
    """
    Annotate the splice table with the marginal psi values.

    Parameters
    ----------
    psi_computation_spec: dict
        The psi computation spec.
    ensembl_version : int
        The ensembl version.
    tissue_id_function : function
        The tissue id function, which maps the tissue name to the key.
    CL_max : int
        The CL max.

    Returns
    -------
    splice_table : pd.DataFrame
        The splice table.
    psi_dump : dict
        The psi pickle dump.
    tissue_keys : list
        The keys.
    bad_genes : pd.DataFrame
        The bad genes splice table.
    """
    splice_table, name_to_ensg, frac_kept = produce_splice_table()
    print(f"Fraction kept: {frac_kept:.2%}")
    tissue_keys, idxs = transcript_data().group_tissue_ids(tissue_id_function)

    ensg_to_psi_dict = construct(
        dict(
            compute_psis_directly=compute_psis_from_junction_data,
            compute_psis_from_annotation_sequence=compute_psis_all,
        ),
        psi_computation_spec,
        gene_ensgs=sorted(name_to_ensg.values()),
        tissue_idxs=idxs,
        ensembl_version=ensembl_version,
    )
    columns, valid_genes, score_data, keys_set = compute_for_all_genes(
        splice_table, name_to_ensg, ensg_to_psi_dict
    )
    for key, value in columns.items():
        splice_table[key] = value

    splice_table, bad_genes = split_splice_table(splice_table, valid_genes, CL_max)

    psi_dump = dict(all_keys=keys_set, psi_values=np.array(score_data))

    return splice_table, tissue_keys, psi_dump, bad_genes


def split_splice_table(splice_table, valid_genes, CL_max):
    """
    Split the splice table into valid and invalid genes.

    Parameters
    ----------
    splice_table : pd.DataFrame
        The splice table.
    valid_genes : np.ndarray
        The valid genes.
    gene_lengths : dict
        The gene lengths.
    CL_max : int
        The CL max.

    Returns
    -------
    splice_table : pd.DataFrame
        The splice table, selected for valid genes.
    bad_genes : pd.DataFrame
        The bad genes.
    """
    # ensure start > CL_max // 2
    valid_genes = valid_genes & (splice_table["start"] >= CL_max // 2)
    # ensure end < gene_length - CL_max // 2
    end_rail = splice_table["chr"].apply(lambda x: gene_lengths[x]) - CL_max // 2
    valid_genes = valid_genes & (splice_table["end"] <= end_rail)
    bad_genes = splice_table[~np.array(valid_genes)]
    splice_table = splice_table[valid_genes].copy()
    return splice_table, bad_genes


def compute_for_all_genes(splice_table, name_to_ensg, ensg_to_psi_dict):
    """
    Compute the splicepoint information for all genes.

    Parameters
    ----------
    splice_table : pd.DataFrame
        The splice table.
    name_to_ensg : dict
        The name to ensg mapping.
    ensg_to_psi_dict : dict
        The ensg to psi dict.

    Returns
    -------
    columns : dict
        The columns to add to the splice table.
    valid_genes : np.ndarray
        The valid genes.
    score_data : list
        The score data.
    keys : list
        The keys.
    """
    columns = dict(
        start=[],
        end=[],
        start_points=[],
        end_points=[],
        start_ids=[],
        end_ids=[],
    )
    score_data = []
    keys = []
    valid_genes = []
    for _, row in splice_table.iterrows():
        psi_d = ensg_to_psi_dict[name_to_ensg[row["name"]]]
        gene_obj = psi_d["gene_obj"]
        columns["start"].append(gene_obj.start)
        columns["end"].append(gene_obj.end)
        # splice site related stuff
        valid, result = construct_for_gene(gene_obj, psi_d, keys, score_data)
        valid_genes.append(valid)
        for key, value in result.items():
            columns[key].append(value)

    valid_genes = np.array(valid_genes)
    return columns, valid_genes, score_data, keys


def construct_for_gene(gene_obj, psi_d, keys_out, score_data_out):
    """
    Construct the splicepoint information for a gene.

    Parameters
    ----------
    gene_obj : Gene
        The gene object.
    psi_d : dict
        The psi dictionary. Contains keys "scores_by_tissue", "sites", and "gene_obj".
    keys_out : list
        The list of keys to append to. Each key is a tuple of (chromosome, strand, site, is_end).
    score_data_out : list
        The list of score data to append to. Each score data is a list of scores for each tissue.

    Returns
    -------
    valid : bool
        Whether the gene is valid.
    result: dict
        The result dictionary. Contains keys "start_points", "end_points", "start_ids", and "end_ids", which
            each map to strings of comma-separated values.
    """
    sp, ep, si, ei = [], [], [], []
    valid = True
    for i, (site, a_or_d) in enumerate(psi_d["sites"]):
        valid = valid and gene_obj.start < site < gene_obj.end
        if a_or_d == "A":
            sp.append(site)
            si.append(len(score_data_out))
        else:
            assert a_or_d == "D"
            ep.append(site)
            ei.append(len(score_data_out))
        keys_out.append(("chr" + gene_obj.contig, gene_obj.strand, site, a_or_d == "D"))
        score_data_out.append(psi_d["scores_by_tissue"][:, i])
    result = dict(start_points=sp, end_points=ep, start_ids=si, end_ids=ei)
    result = {k: ",".join(map(str, v)) for k, v in result.items()}
    return valid, result


# https://genome.ucsc.edu/cgi-bin/hgTracks?chromInfoPage=

gene_lengths = {
    "chr1": 248956422,
    "chr2": 242193529,
    "chr3": 198295559,
    "chr4": 190214555,
    "chr5": 181538259,
    "chr6": 170805979,
    "chr7": 159345973,
    "chr8": 145138636,
    "chr9": 138394717,
    "chr10": 133797422,
    "chr11": 135086622,
    "chr12": 133275309,
    "chr13": 114364328,
    "chr14": 107043718,
    "chr15": 101991189,
    "chr16": 90338345,
    "chr17": 83257441,
    "chr18": 80373285,
    "chr19": 58617616,
    "chr20": 64444167,
    "chr21": 46709983,
    "chr22": 50818468,
    "chrX": 156040895,
    "chrY": 57227415,
    "chrM": 16569,
}
