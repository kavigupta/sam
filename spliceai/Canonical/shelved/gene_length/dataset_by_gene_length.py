import h5py
import numpy as np
import tqdm.auto as tqdm
from permacache import permacache

from modular_splicing.dataset.dataset_aligner import DatasetAligner
from modular_splicing.data_for_experiments.standardized_sample import (
    standardized_sample,
)


@permacache("dataset/dataset_by_gene_length/dataset_by_gene_length_2")
def dataset_by_gene_length(
    *,
    dataset_path,
    datafile_path,
    cl_max,
    data_sl,
    sl,
    scan_amount=50_000,
    gene_length_threshold=10_000,
):
    xs, ys, indices = standardized_sample(
        dataset_path,
        amount=scan_amount,
        get_motifs=True,
        datapoint_extractor_spec=dict(
            type="BasicDatapointExtractor",
            rewriters=[
                dict(
                    type="AdditionalChannelDataRewriter",
                    out_channel=["inputs", "motifs"],
                    data_provider_spec=dict(type="index_tracking"),
                ),
            ],
        ),
        sl=sl,
        cl=cl_max,
    )
    indices = np.array([i[0][0] for i in indices])

    aligner = DatasetAligner(
        dataset_path=dataset_path, datafile_path=datafile_path, sl=data_sl
    )
    gene_idxs = [aligner.get_gene_idx(*idx)[0] for idx in tqdm.tqdm(indices)]
    with h5py.File(datafile_path, "r") as f:
        lengths = np.array([len(f["SEQ"][gi]) for gi in tqdm.tqdm(gene_idxs)])
        lengths -= cl_max
    short_mask = lengths < gene_length_threshold
    [short_idxs] = np.where(short_mask)
    [long_idxs] = np.where(~short_mask)
    assert short_idxs.size <= long_idxs.size
    long_idxs = long_idxs[: short_idxs.size]

    short = dict(xs=xs[short_idxs], ys=ys[short_idxs], indices=indices[short_idxs])
    long = dict(xs=xs[long_idxs], ys=ys[long_idxs], indices=indices[long_idxs])
    return short, long
