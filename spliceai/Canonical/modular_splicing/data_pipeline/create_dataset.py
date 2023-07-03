import os
import h5py
import numpy as np
import tqdm.auto as tqdm

from .chunk_iteration_order import dataset_indices_generator
from .reprocess_data import create_datapoints

from modular_splicing.utils.genome_statistics import gene_at_richness


def create_dataset(
    *,
    datafile_path,
    dataset_path,
    SL,
    CL_max,
    length_filter=None,
    at_richness_filter=None,
    mode="one-hot",
):
    """
    Create a dataset from a datafile. The dataset is a .h5 file with the
    following structure:
    - Xi: a matrix of shape (n, SL, 4) containing the RNA sequence
    - Yi: a matrix of shape (1, n, 3) containing the splicing outcomes.

    Arguments:
    datafile_path: path to the datafile produced by create_datafile
    dataset_path: path to the dataset to be created
    SL: sequence length
    CL_max: maximum CL to allow
    length_filter: if not None, only keep sequences of this length
    at_richness_filter: if not None, only keep sequences with this AT richness
    """

    h5f = h5py.File(datafile_path, "r")

    STRAND = h5f["STRAND"][:]
    TX_START = h5f["TX_START"][:]
    TX_END = h5f["TX_END"][:]
    JN_START = h5f["JN_START"][:]
    JN_END = h5f["JN_END"][:]
    if "SEQ" in h5f:
        SEQ = h5f["SEQ"][:]
    else:
        SEQ = np.array([None] * len(STRAND))
    h5f.close()

    mask = None

    if length_filter is not None:
        lengths = np.array([len(x) - CL_max for x in SEQ])
        mask = length_filter(lengths)

    if at_richness_filter is not None:
        assert mask is None
        at_richness = np.array(
            [
                gene_at_richness(datafile_path, index, cl_max=10_000)
                for index in range(len(SEQ))
            ]
        )
        mask = at_richness_filter(at_richness)

    if mask is not None:
        STRAND = STRAND[mask]
        TX_START = TX_START[mask]
        TX_END = TX_END[mask]
        JN_START = JN_START[mask]
        JN_END = JN_END[mask]
        SEQ = SEQ[mask]

    h5f2 = h5py.File(dataset_path, "w")

    for i, specific_splice_indices in tqdm.tqdm(
        list(dataset_indices_generator(SEQ.shape[0]))
    ):
        # Each dataset has CHUNK_SIZE genes
        X_batch = []
        Y_batch = [[] for t in range(1)]

        for idx in specific_splice_indices:
            X, Y = create_datapoints(
                SEQ[idx],
                STRAND[idx],
                TX_START[idx],
                TX_END[idx],
                JN_START[idx],
                JN_END[idx],
                SL=SL,
                CL_max=CL_max,
                mode=mode,
            )
            if X is not None:
                X_batch.extend(X)
            for t in range(1):
                Y_batch[t].extend(Y[t])

        X_batch = np.asarray(X_batch).astype("int8")
        for t in range(1):
            Y_batch[t] = np.asarray(Y_batch[t])
            if mode != "external-indices":
                Y_batch[t] = Y_batch[t].astype("int8")

        h5f2.create_dataset("X" + str(i), data=X_batch)
        h5f2.create_dataset("Y" + str(i), data=Y_batch)

    h5f2.close()


def produce_datasets_from_datafile(*, data_path_folder, segment_chunks, CL_max, SL):
    """
    Produce datasets from datafiles.

    Parameters
    ----------
    data_path_folder : str
        Path to folder containing datafiles.
    segment_chunks : list of tuples
        List of tuples of segment and chunk.
    CL_max : int
        Maximum chunk length.
    SL : int
        Sequence length.
    """
    outfiles = {
        (segment, chunk): f"{data_path_folder}dataset_{segment}_{chunk}.h5"
        for segment, chunk in segment_chunks
    }

    if all(os.path.exists(outfile) for outfile in outfiles.values()):
        print("Datasets already exist, skipping creation.")
        return

    for segment, chunk in segment_chunks:
        create_dataset(
            datafile_path=f"{data_path_folder}datafile_{segment}_{chunk}.h5",
            dataset_path=outfiles[segment, chunk],
            SL=SL,
            CL_max=CL_max,
            mode="external-indices",
        )
