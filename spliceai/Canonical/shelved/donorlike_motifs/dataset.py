from types import SimpleNamespace

import tqdm.auto as tqdm

import numpy as np
import torch

from shelved.donorlike_motifs.utils import load_sparsest
from modular_splicing.dataset import basic_dataset
from modular_splicing.utils.run_batched import run_batched

from modular_splicing.models.modules.lssi_in_model import LSSI_MODEL_THRESH


def standard_data(*, total=50_000, version=1):
    dset = get_dset(total=total, require_ad=True)
    xs = np.array([x for x, _ in dset])
    ys = np.eye(3, dtype=np.bool)[np.array([y for _, y in dset])][:, :, 1:]
    donors = ys[:, :, -1]
    with torch.no_grad():
        splicepoint_model = load_sparsest("msp-141d1_1").splicepoint_model
        plausible_splicepoints = run_batched(
            splicepoint_model.forward_just_splicepoints, xs, 32, pbar=tqdm.tqdm
        )
        plausible_splicepoints = plausible_splicepoints[:, 200:-200]
        plausible_donors = plausible_splicepoints[:, :, -1]
    plausible_donors_above_thresh = plausible_donors > LSSI_MODEL_THRESH
    plausible_donors_topk = plausible_donors > np.quantile(
        plausible_donors, 1 - donors.mean()
    )
    return SimpleNamespace(
        xs=xs,
        ys=ys,
        donors=donors,
        plausible_donors_above_thresh=plausible_donors_above_thresh,
        plausible_donors_topk=plausible_donors_topk,
    )


def get_dset(
    total=1000,
    sl=1000,
    require_ad=True,
    pbar=tqdm.tqdm,
    dataset="train_all",
    cl=400,
):
    raise RuntimeError("obsolete")
    data = basic_dataset(f"dataset_{dataset}.h5", cl, 10_000, sl=sl)
    dset = []
    pbar = pbar(total=total)
    for xs, ys in iter(data):
        if require_ad:
            if not ((ys == 1).sum() and (ys == 2).sum()):
                continue
        dset.append((xs, ys))
        pbar.update()
        if len(dset) == total:
            pbar.close()
            break
    return dset
