import json
import shutil
import subprocess
import os
import re
import tarfile
import tempfile

from permacache import permacache
import tqdm.auto as tqdm
import pandas as pd

common = [
    "geneSymbol",
    "chr",
    "strand",
    "PValue",
    "FDR",
    "IncLevelDifference",
    "IJC_SAMPLE_1",
    "IJC_SAMPLE_2",
    "SJC_SAMPLE_1",
    "SJC_SAMPLE_2",
]

alternate_splice_site = [
    "longExonStart_0base",
    "longExonEnd",
    "shortES",
    "shortEE",
    "flankingES",
    "flankingEE",
    *common,
]
cols = {
    "SE": [
        "exonStart_0base",
        "exonEnd",
        "upstreamES",
        "upstreamEE",
        "downstreamES",
        "downstreamEE",
        *common,
    ],
    "A3SS": alternate_splice_site,
    "A5SS": alternate_splice_site,
}

MATCHER = (
    r"(?P<motif_name>[^-]+)-(?P<experiment>[^-]+)-(?P<cell_line>[^-]+)"
    + r"/MATS_output"
    + r"/(?P<category>[^-]+).MATS.JunctionCountOnly.txt"
)


def read_file(path, filter_columns=True):
    """
    Read the gzip file at `path` into a pandas dataframe.

    See `knockdown_dataset` for more details on what the return type looks like.
    """
    all_metadata = []
    all_data = []
    with tarfile.open(path, "r:gz") as tar:
        for tarinfo in tar.getmembers():
            if tarinfo.isdir():
                continue
            if os.path.basename(tarinfo.name)[0] == ".":
                continue
            metadata = re.match(MATCHER, tarinfo.name).groupdict()
            data = pd.read_csv(tar.extractfile(tarinfo.name), sep="\t")
            all_metadata.append(metadata)
            all_data.append(data)
    metadata_without_category = [
        {k: v for k, v in meta.items() if k != "category"} for meta in all_metadata
    ]
    for x in metadata_without_category:
        assert x == metadata_without_category[0]
    by_category = {
        meta["category"]: data
        for meta, data in zip(all_metadata, all_data)
        if meta["category"] in ("A3SS", "A5SS", "SE")
    }
    if filter_columns:
        for col in cols:
            by_category[col] = by_category[col][cols[col]]
    return {
        **metadata_without_category[0],
        **by_category,
    }


@permacache(
    "validation/knockdown_dataset/run_download_3", key_function=dict(folder=None)
)
def run_download(urls, folder):
    """
    Run a download of all the urls. See `knockdown_dataset` for more details.
    """
    if folder is None:
        folder = tempfile.mkdtemp()
    os.makedirs(folder)
    for u in tqdm.tqdm(urls):
        subprocess.check_call(["curl", "-OJL", u], cwd=folder)
    sort_RNAi_tarfiles(folder)
    path = f"{folder}/rMATS"
    results = []
    for file in tqdm.tqdm(os.listdir(path)):
        results.append(read_file(os.path.join(path, file)))
    return pd.DataFrame(results)


def knockdown_dataset(
    origin_path="../data/encode-project-rna-maps/files_RNAi_hg19.txt", folder=None
):
    """
    Produce a dataset of knockdown experiments.

    Output is a pandas dataframe with the following columns:
        motif_name: the name of the motif
        experiment: the name of the experiment type
        cell_line: the name of the cell line
        A3SS: a dataframe of A3SS events
        A5SS: a dataframe of A5SS events
        SE: a dataframe of SE events

    Each of the A3SS, A5SS, and SE dataframes has the following columns:
        geneSymbol: the name of the gene
        chr: the chromosome
        strand: the strand
        PValue: the p-value
        FDR: the false discovery rate
        IncLevelDifference: the difference in inclusion level
        IJC_SAMPLE_1: the number of junctions in sample 1
        IJC_SAMPLE_2: the number of junctions in sample 2
        SJC_SAMPLE_1: the number of skipped junctions in sample 1
        SJC_SAMPLE_2: the number of skipped junctions in sample 2
    """
    with open(origin_path) as f:
        urls = f.read().split("\n")[1:]
        urls = [x for x in urls if x]
    return run_download(urls, folder)


dataset_for_cell_line_cache = {}


def dataset_for_cell_line(cell_line, **kwargs):
    """
    Produce a dataset for a given cell line.

    See `knockdown_dataset` for more details.
    """
    key = json.dumps(dict(cell_line=cell_line, **kwargs))
    if key not in dataset_for_cell_line_cache:
        dataset_for_cell_line_cache[key] = _dataset_for_cell_line_direct(
            cell_line, **kwargs
        )
    return dataset_for_cell_line_cache[key]


def _dataset_for_cell_line_direct(cell_line, **kwargs):
    assert isinstance(cell_line, str)
    dset = knockdown_dataset(**kwargs)
    dset = dset[dset["cell_line"] == cell_line]
    return dset.set_index("motif_name")


def check_output_type(tarpath, basepath):
    """
    Check the output type of the tarfile at `tarpath`.
    """
    contents = []
    with tarfile.open(f"{basepath}/{tarpath}.tar.gz", "r:gz") as tar:
        for tarinfo in tar.getmembers():
            contents.append("/".join(tarinfo.name.split("/")[1:]))
    if "MATS_Norm_output" in contents:
        output_type = "rMATS_norm"
    elif "MATS_output" in contents:
        output_type = "rMATS"
    else:
        assert "miso" in "".join(contents)
        output_type = "MISO"
    return output_type


def sort_RNAi_tarfiles(dirpath):
    """
    Sort the RNAi tarfiles in `dirpath` into subdirectories.
    """
    filelist = os.listdir(dirpath)
    miso_path = f"{dirpath}/MISO"
    norm_path = f"{dirpath}/rMATS_norm"
    rmats_path = f"{dirpath}/rMATS"
    os.makedirs(miso_path)
    os.makedirs(norm_path)
    os.makedirs(rmats_path)
    mv_dir = {"MISO": miso_path, "rMATS": rmats_path, "rMATS_norm": norm_path}

    for f in tqdm.tqdm(filelist):
        file_parts = f.split(".")
        root = file_parts[0]
        ext = ".".join(file_parts[1:])
        if ext != "tar.gz":
            continue
        folder_type = check_output_type(root, dirpath)
        newdir = mv_dir[folder_type]
        shutil.move(f"{dirpath}/{f}", f"{newdir}/{f}")
