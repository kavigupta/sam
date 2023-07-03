from tempfile import NamedTemporaryFile
import tqdm.auto as tqdm

from permacache import permacache

import h5py
import numpy as np
import pandas as pd
import scipy.sparse

from modular_splicing.dataset.datafile_object import SpliceAIDatafile
from modular_splicing.data_pipeline.create_dataset import create_dataset
from modular_splicing.dataset.additional_data import AdditionalData
from modular_splicing.utils.sparse_tensor import pad_sparse_motifs_with_cl
from modular_splicing.utils.construct import construct


def get_branchpoints():
    return pd.read_csv(
        "../data/branchpoints/FileS2.bed",
        sep="\t",
        names=["chrom", "chromStart", "chromEnd", "name", "score", "strand"],
    )


def write_branch_datafile(in_path, out_path):
    data = SpliceAIDatafile.load(in_path)
    classified = data.classify_bed_rows(get_branchpoints())
    with h5py.File(out_path, "w") as branch_datafile:
        for k in data.datafile.keys():
            if k == "SEQ":
                continue
            branch_datafile[k] = data.datafile[k][:]
        for idx in tqdm.trange(len(data.datafile["SEQ"])):
            if classified[idx]:
                frame = pd.DataFrame(classified[idx])
                starts, ends = frame.chromStart, frame.chromEnd
            else:
                starts, ends = [], []
            starts, ends = [
                np.array([("".join(str(x) + "," for x in xs)).encode("ascii")])
                for xs in (starts, ends)
            ]
            branch_datafile["JN_START"][idx], branch_datafile["JN_END"][idx] = (
                starts,
                ends,
            )


@permacache("branch_sites/pipeline/branch_sites_dataset")
def branch_site_dataset(datafile_path, *, CL_max=10_000, SL=5000):
    with NamedTemporaryFile() as branch_datafile_path, NamedTemporaryFile() as branch_dataset_path:
        write_branch_datafile(datafile_path, branch_datafile_path.name)
        create_dataset(
            datafile_path=branch_datafile_path.name,
            dataset_path=branch_dataset_path.name,
            CL_max=CL_max,
            SL=SL,
        )
        with h5py.File(branch_dataset_path.name, "r") as result:
            out = {}
            for i in tqdm.trange(len(result)):
                key = f"Y{i}"
                if key not in result:
                    break
                out[i] = scipy.sparse.csr_matrix(result[key][0, :, :, -1])
    return out


def expand_branch_sites(datafile, params, i, j):
    dset = expand_all_branch_sites(datafile, params, i)
    dset = dset.getrow(j).T
    dset = dset.toarray()
    return dset


def expand_all_branch_sites(datafile, params, i):
    dset = branch_site_dataset(datafile_path=datafile, **params)
    dset = pad_sparse_motifs_with_cl(dset[i])
    return dset


@permacache("branch_sites/pipeline/compute_mask_2")
def compute_mask(
    *, datafile_path, dataset_path, branch_site_params, num_branch_site_filter, i
):
    with h5py.File(dataset_path, "r") as f:
        y = f[f"Y{i}"][0]

    cl = branch_site_params.get("CL_max", 10_000)
    y_flat = y[:, :, 1:].reshape(-1, 2)
    indices, is_don = np.where(y_flat)
    indices, is_don = clean_splice_indices(indices, is_don)
    assert (is_don[:-1] != is_don[1:]).all() and is_don[0] and indices.size % 2 == 0
    intron_starts, intron_ends = indices[::2], indices[1::2] + 1
    exon_starts = np.concatenate([[0], intron_ends])
    exon_ends = np.concatenate([intron_starts, [y_flat.shape[1]]])
    exon_mids = (exon_starts + exon_ends) // 2

    sites = flat_sites(datafile_path, branch_site_params, i, cl)

    assert sites.shape[0] == y_flat.shape[0]
    [sites] = np.where(sites)
    count_introns = (
        (intron_starts[None] <= sites[:, None]) & (sites[:, None] < intron_ends[None])
    ).sum(0)

    valid_introns = construct(
        dict(
            at_least=lambda x, value: x >= value,
            exactly=lambda x, value: x == value,
        ),
        num_branch_site_filter,
        x=count_introns,
    )
    mask = np.zeros(y_flat.shape[0], dtype=np.bool)
    for intron_idx in np.where(valid_introns)[0]:
        # from the previous exon's middle (same index) to the next exon's middle (next index)
        mask[exon_mids[intron_idx] : exon_mids[intron_idx + 1]] = True
    return mask.reshape(y.shape[:2])


def flat_sites(datafile_path, branch_site_params, i, cl):
    sites = expand_all_branch_sites(datafile_path, params=branch_site_params, i=i)
    sites = sites.toarray()
    sites = sites[:, cl // 2 : sites.shape[1] - cl // 2]
    sites = sites.flatten()
    return sites


def clean_splice_indices(indices, is_don):
    ri, rd = [], []
    bad = 0
    prev = 0
    for i, d in zip(indices, is_don):
        if d == prev:
            bad += 1
            continue
        prev = d
        ri.append(i)
        rd.append(d)
    ri, rd = np.array(ri), np.array(rd)
    assert bad <= 1
    return ri, rd


class BranchSiteAdditionalData(AdditionalData):
    """
    Wrapper around expand_branch_sites that is compatible with the
    AdditionalData interface. Runs expand_branch_sites with the relevant
    datafile.

    Parameters:
    -----------
    datafiles : dict
        Dictionary mapping "True" and "False" to the paths of the
        training and validation datafiles, respectively.

        Keys are strings for json compatibility.
    branch_site_dataset_params : dict
        Parameters to pass to expand_branch_sites.
    """

    def __init__(self, datafiles, branch_site_dataset_params={}):
        self.datafiles = datafiles
        self.branch_site_dataset_params = branch_site_dataset_params

    def compute_additional_input(self, original_input, path, i, j):
        is_train = self.classify_path(path)
        return expand_branch_sites(
            self.datafiles[str(is_train)], self.branch_site_dataset_params, i, j
        )


class BranchSiteMaskAdditionalData(AdditionalData):
    """
    Wrapper around `compute_mask` that is compatible with the `AdditionalData` interface.

    See `compute_mask` for more details.
    """

    def __init__(
        self,
        datafiles,
        branch_site_dataset_params={},
        num_branch_site_filter=dict(type="at_least", value=1),
    ):
        self.datafiles = datafiles
        self.branch_site_dataset_params = branch_site_dataset_params
        self.num_branch_site_filter = num_branch_site_filter

    def compute_additional_input(self, original_input, path, i, j):
        raise RuntimeError("Not implemented")

    def compute_additional_output(self, original_input, path, i, j, cl_max):
        is_train = self.classify_path(path)
        return compute_mask(
            datafile_path=self.datafiles[str(is_train)],
            dataset_path=path,
            branch_site_params=self.branch_site_dataset_params,
            num_branch_site_filter=self.num_branch_site_filter,
            i=i,
        )[j]
