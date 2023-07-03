from permacache import permacache, stable_hash

from modular_splicing.data_pipeline.spliceai_data_processing_pipeline import (
    create_dataset_from_splice_table_no_sequence,
)
from modular_splicing.utils.arrays import Sparse


@permacache(
    "dataset/alternative_dataset/get_alternative_dataset_2",
    key_function=dict(canonical_data=stable_hash, data_txt_files=stable_hash),
)
def merged_dataset(
    *,
    canonical_data,
    data_txt_files,
    files_for_gene_intersection,
    data_segment_chunks_to_use,
    ref_genome_path,
):
    """
    Produces a merged dataset  out of the given splice tables.

    Parameters
    ----------
    canonical_data, data_txt_files: the first two outputs of `splice_tables_on_common_genes`
    files_for_gene_intersection: the files to use for gene intersection, i.e.,
        drop all genes not in all of these files
    data_segment_chunks_to_use: list of tuples, e.g., [("train", "all"), ("test", "0")]
    ref_genome_path: path to the reference genome fa file

    Returns (xs, ys_each)
    -------
    xs: dict of Xi -> np.ndarray[N, L + CL_max, 4]
        the input values for each data segment
    ys_each: dict of data_key -> (dict of Yi -> Sparse[N, L])
        the output values for each data segment
    """

    splice_tables_each = align_with_intersection(
        canonical_data,
        data_txt_files,
        files_for_gene_intersection=files_for_gene_intersection,
    )
    xs = {}
    ys_each = {ch: {} for ch in data_segment_chunks_to_use}
    for i, k in enumerate(splice_tables_each):
        res = create_dataset_from_splice_table_no_sequence(
            ref_genome=ref_genome_path,
            splice_table_frame=splice_tables_each[k],
            load_file=lambda dataset: load_x_and_y_sparse(dataset, i == 0),
            data_segment_chunks_to_use=data_segment_chunks_to_use,
            CL_max=10_000,
            SL=5000,
            include_seq=i == 0,
        )
        for ch in res:
            x, y = res[ch]
            if i == 0:
                xs[ch] = x
            ys_each[ch][k] = y
    return xs, ys_each


def load_x_and_y_sparse(dataset, get_x):
    """
    Loads the data from the given dataset, with the outputs being sparse matrices.

    Parameters
    ----------
    dataset: the dataset to load from
    get_x: whether to get the x values

    Returns
    -------
    x: np.ndarray[N, L + CL_max, 4]
        the input values. If get_x is False, this is None
    y: Sparse[N, L]
        the output values
    """
    if get_x:
        xres = {k: dataset[k][:].argmax(-1) for k in dataset if k.startswith("X")}
    else:
        xres = None
    yres = {}
    for k in dataset:
        if not k.startswith("Y"):
            assert k.startswith("X")
            continue
        assert dataset[k].shape[0] == 1
        yres[k] = Sparse.of(dataset[k][0].argmax(-1))
    return xres, yres


def align_with_intersection(
    canonical_data, data_txt_files, *, files_for_gene_intersection
):
    """
    Align the given splice tables with the intersection over the genes.

    Parameters
    ----------
    canonical_data, data_txt_files: the first two outputs of `splice_tables_on_common_genes`
    files_for_gene_intersection: the files to use for gene intersection, i.e.,
        drop all genes not in all of these files

    Returns
    -------
    splice_tables_each: dict of data_key -> pd.DataFrame
        updated splice tables for each key. Only include genes in the intersection
    """
    common_genes = intersect_all(
        [data_txt_files[k].Name for k in files_for_gene_intersection]
    )
    common_genes_table = canonical_data[
        canonical_data.Name.apply(lambda x: x in common_genes)
    ].copy()
    return {
        k: gather_exons_for_index(common_genes_table, data_txt_files[k])
        for k in data_txt_files
    }


def intersect_all(elements):
    """
    Intersect a list of sets.
    """
    elements = list(elements)
    if len(elements) == 0:
        raise ValueError("Empty list")
    start = set(elements[0])
    for e in elements[1:]:
        start = start & set(e)
    return start


def gather_exons_for_index(metadata_frame, data_frame):
    """
    metadata_frame: the frame to use for all the gene metadata
    data_frame: the frame to use for the exons
    """
    metadata_frame = metadata_frame.copy()
    frame_values = dict(zip(data_frame.Name, zip(data_frame.sstarts, data_frame.sends)))
    sstarts_sends = metadata_frame.Name.apply(lambda x: frame_values.get(x, ("", "")))
    metadata_frame.sstarts, metadata_frame.sends = [
        sstarts_sends.apply(lambda x: x[idx]) for idx in range(2)
    ]
    return metadata_frame
