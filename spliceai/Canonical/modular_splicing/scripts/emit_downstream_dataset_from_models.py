import os
import pickle

import tqdm.auto as tqdm

import h5py
import numpy as np
import torch

from modular_splicing.models.modules.lssi_in_model import BothLSSIModels
from modular_splicing.utils.arrays import Sparse
from modular_splicing.utils.run_batched import run_batched

from modular_splicing.utils.io import load_model

import scipy.sparse

from modular_splicing.models_for_testing.list import FM, AM


def populate_models_directory(models_directory):
    try:
        os.makedirs(models_directory)
    except FileExistsError:
        pass
    for model in FM.non_binarized_models() + AM.binarized_models():
        print(model)
        torch.save(model.model, os.path.join(models_directory, model.name))


def compute_downstream_dataset(models_directory, dataset_path, idx, num_motifs=79):
    """
    Compute the downstream dataset for the given models.

    models_directory: path to a directory containing the models to be used
    dataset_path: path to the dataset to be used
    idx: index of the chunk in the dataset to be used

    returns: a dictionary with keys
        - "X" (N, L + CL_max, 4): the input data
        - "Y" (N, L, 3): the output data
        - "S" (N, L + CL_max, 2): the LSSI outcomes
        - "M_{n}" sparse(N, L + CL_max, M): for each model n, the model's motif outputs
    """
    spm = BothLSSIModels(
        acceptor="splicepoint-models/acceptor.m", donor="splicepoint-models/donor2.m"
    ).cuda()
    with h5py.File(dataset_path, "r") as in_f:
        x, y = in_f[f"X{idx}"][:], in_f[f"Y{idx}"][0]
    s = run_batched(
        spm.forward_just_splicepoints, x.astype(np.float32), 256, pbar=tqdm.tqdm
    )
    result = {"X": x, "Y": y, "S": s}

    for n in os.listdir(models_directory):
        model_path = os.path.join(models_directory, n)
        assert os.path.isfile(model_path)
        _, mod = load_model(model_path)
        mod = mod.eval()
        m = run_batched(
            lambda x: mod(x, only_motifs=True)["post_sparse_motifs_only"],
            x.astype(np.float32),
            32,
            pbar=tqdm.tqdm,
        )
        m = m[:, :, :num_motifs]
        result[f"M_{n}"] = [scipy.sparse.coo_matrix(x) for x in m]
    return result


def output_downstream_dataset(models_directory, dataset_path, idx, out_folder):
    """
    Output the given downstream dataset to the given folder.

    See compute_downstream_dataset for the meaning of the arguments and the output format.

    Results are dumped to out_folder/idx.pkl. If this file exists, we skip the computation.
    """
    out_path = os.path.join(out_folder, f"{idx}.pkl")
    if os.path.exists(out_path):
        return

    try:
        os.makedirs(out_folder)
    except FileExistsError:
        pass
    with open(out_path, "wb") as out_f:
        pickle.dump(
            compute_downstream_dataset(models_directory, dataset_path, idx), out_f
        )


if __name__ == "__main__":
    models_directory = "/mnt/md0/models-for-dataset-for-downstream-shorter-donor"
    out = "/mnt/md0/downstream-dataset-shorter-donor"
    populate_models_directory(models_directory)
    # usage example. You can use indices 0-132 for training and 0-15 for testing.
    for idx in range(133):
        print("train", idx)
        output_downstream_dataset(
            models_directory,
            # the dataset
            "dataset_train_all.h5",
            # the index
            idx,
            # the output folder
            os.path.join(out, "train"),
        )
    for idx in range(16):
        print("test", idx)
        output_downstream_dataset(
            models_directory,
            # the dataset
            "dataset_test_0.h5",
            # the index
            idx,
            # the output folder
            os.path.join(out, "test"),
        )
