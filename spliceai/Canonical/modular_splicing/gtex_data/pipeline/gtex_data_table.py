from functools import lru_cache
import os
import re

import h5py
from more_itertools import chunked
import numpy as np
import pandas as pd
import tqdm.auto as tqdm

GENE_EXPR_GCT = (
    "../data/gtex_junctions/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_reads.gct"
)
GENE_EXPR_H5 = "../data/gtex_junctions/gene_reads.h5"
EXON_JUNCTION_GCT = (
    "../data/gtex_junctions/GTEx_Analysis_2017-06-05_v8_STARv2.5.3a_junctions.gct"
)
EXON_JUNCTION_H5 = "../data/gtex_junctions/exon_junctions.h5"

TRANSCRIPT_GCT = (
    "../data/gtex_junctions/GTEx_Analysis_2017-06-05_v8_RSEMv1.3.0_transcript_tpm.gct"
)
TRANSCRIPT_H5 = "../data/gtex_junctions/transcript_tpm.h5"

CHUNK_SIZE = 100


def gene_expr_data():
    """
    Returns a FeatureDataBySample object containing the gene expression data.
    """
    return FeatureDataBySample(GENE_EXPR_GCT, GENE_EXPR_H5)


def exon_junction_data():
    """
    Returns a FeatureDataBySample object containing the exon junction data.
    """
    return FeatureDataBySample(EXON_JUNCTION_GCT, EXON_JUNCTION_H5)


def transcript_data():
    """
    Returns a FeatureDataBySample object containing the transcript data.
    """
    return FeatureDataBySample(TRANSCRIPT_GCT, TRANSCRIPT_H5, dtype=float)


class FeatureDataBySample:
    """
    A class for accessing the gene expression, exon junction, and transcript data
        from the GTEx project.

    You can index into i to get the ith row of the data. The row is returned as a
        numpy array.

    Parameters
    ----------
    input_gct : str
        The path to the input gct file.
    output_h5 : str
        The path to the output h5 file. Created if it does not exist.
    dtype : type
        The dtype of the data. Defaults to int.

    Attributes
    ----------
    The following fields correspond to the columns in the gct file.
        sample_ids : list of str
            The sample ids.
        donor_ids : list of str
            The donor ids.
        aliquots : list of str
            The aliquot ids.
        tissue_ids : list of str
            The tissue ids.

    The following fields correspond to the rows in the gct file.
        meta_0 : list of str
            The first column of the gct file.
        meta_1 : list of str
            The second column of the gct file.

    """

    def __init__(self, input_gct, output_h5, **kwargs):
        if not os.path.exists(output_h5):
            construct_gene_expr_data(input_gct, output_h5, **kwargs)
        self._file = h5py.File(output_h5, "r")
        self.sample_ids = [
            x.decode("ascii").strip() for x in self._file["sample_ids"][:]
        ]
        self.donor_ids, _, self.aliquots = zip(
            *[
                re.match(r"GTEX-([0-9A-Z]{4,5})-(.*)-(SM.*)", x).groups()
                for x in self.sample_ids
            ]
        )
        tim = tissue_id_maps()
        self.tissue_ids = [tim[x] for x in self.sample_ids]

        self.meta_0 = [x.decode("ascii") for x in self._file["meta_0"][:]]
        self.meta_1 = [x.decode("ascii") for x in self._file["meta_1"][:]]

    def __getitem__(self, key):
        i = key // CHUNK_SIZE
        j = key % CHUNK_SIZE
        return self._file[f"E{i}"][j]

    def __len__(self):
        return len(self.meta_0)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._file.close()

    def group_tissue_ids(self, group_key):
        """
        Get the tissue ids grouped by a key.

        Parameters
        ----------
        group_key : function
            A function that takes a tissue id and returns a key.

        Returns
        -------
        keys : list of str
            The keys.
        idxs : list of list of int
            The indices of the tissue ids corresponding to each key.
        """
        keys = [group_key(t) for t in self.tissue_ids]
        keys_set = sorted(set(keys))
        idxs = [np.where([k2 == k for k2 in keys])[0] for k in keys_set]
        return keys_set, idxs


def construct_gene_expr_data(input_gct, output_h5, **kwargs):
    """
    Convert the gene expression data from gct to h5 format.
    """
    with open(input_gct) as f:
        next(f), next(f)
        content = list(f)
    _, _, *names = content[0].split("\t")

    with h5py.File(output_h5, "w") as f:
        genes = []
        for i, elements in enumerate(
            chunked(read_from_file(content[1:], genes, **kwargs), CHUNK_SIZE)
        ):
            f.create_dataset(f"E{i}", data=np.array(elements))
        meta_0, meta_1 = zip(*genes)
        f.create_dataset("meta_0", data=[x.encode("ascii") for x in meta_0])
        f.create_dataset("meta_1", data=[x.encode("ascii") for x in meta_1])
        f.create_dataset("sample_ids", data=[x.encode("ascii") for x in names])


@lru_cache(None)
def tissue_id_maps():
    """
    Loads a map from sample id to tissue id.
    """
    sample_table = pd.read_csv(
        "../data/gtex_junctions/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt",
        sep="\t",
    )
    return dict(zip(sample_table.SAMPID, sample_table.SMTSD))


def read_from_file(content, meta, dtype=int):
    """
    Reads the gene expression data from the gct file.

    Parameters
    ----------
    content : list of str
        The lines of the gct file.
    meta : list of (str, str)
        A list to append the meta data to.
    dtype : type
        The dtype of the data. Defaults to int.

    Yields
    ------
    np.array
        The data for the next row.
    """
    for line in tqdm.tqdm(content):
        first, second, *line = line.split("\t")
        meta.append((first, second))
        yield np.array([dtype(x) for x in line])
