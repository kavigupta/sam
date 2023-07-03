from datetime import datetime
import numpy as np
import torch
import os
import time

from constants import get_args

from spliceai_torch import (
    SpliceAI,
    load_model, 
    save_model,
    SpliceAIDataset,
    evaluate_model,
)

from load_psam_motif import read_motifs
from map_protein_name import get_map_protein

from hyperparams import get_hparams

from load_rbns_motif import (
    rbns_motifs,
    rbns_motifs_for, 
)


def main(
    window, 
    CL_max, 
    l,
    lr, 
    bs, 
    motifs, 
    sparsity,
    data_dir,
    organism,
    SL, 
    path,
    n_epochs,
    attr,
    ):

    torch.set_num_threads(1)
    # attr = f"{sparsity}_{bs}_{lr}"

    skip_to_step, m = load_model(path, attr)
    # print(m)

    deval = SpliceAIDataset.of(
        data_dir + "/" + organism + "_" + "dataset" + "_" + "test" + "_" + "all" + ".h5",
        cl=window,
        cl_max=CL_max,
        sl=SL,
    )

    evaluate_model(m, deval, bs=bs, limit=float("inf"), quiet=False)


def run():
    args = get_args()

    main(
    window=args.window, 
    CL_max=args.CL_max, 
    l=args.l,
    lr=args.lr, 
    bs=args.bs, 
    motifs=rbns_motifs(psam_only=args.psam_only),
    sparsity=args.sparsity,
    data_dir=args.data_dir,
    organism=args.organism,
    SL=args.SL,
    path=args.model_path,
    n_epochs=args.n_epochs,
    attr=args.attr,
    )


if __name__ == "__main__":
    run()