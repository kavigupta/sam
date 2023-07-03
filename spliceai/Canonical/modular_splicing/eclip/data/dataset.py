import h5py
import numpy as np

from permacache import permacache, drop_if_equal

import scipy.sparse
from modular_splicing.dataset.additional_data import AdditionalData
from modular_splicing.utils.intron_exon_annotation import to_intron_exon_annotation


from .pipeline import eclip_dataset_with_spliceai_pipeline
from ..test_motifs.names import peak_names


def get_eclip_dataset(is_train, *, replicate_category, dataset_pipeline_mode):
    """
    Gets the eclip dataset. See the `eclip_dataset_with_spliceai_pipeline` function
    for more details.

    Parameters
    ----------
    is_train: whether or not to get the training dataset
    replicate_category: which replicate category to get
    dataset_pipeline_mode: which dataset pipeline mode to use

    Returns
    -------
        a list of data chunks, each of which is a list of sparse matrices
        representing a tensor of shape (N, 2L, M), where M is the number of motifs
        and it is padded out with 0s on either side. Here, 1 == start of motif, 2 == end of motif
    """
    if dataset_pipeline_mode == "with_spliceai_pipeline":
        _, _, result = eclip_dataset_with_spliceai_pipeline(
            replicate_category=replicate_category,
            is_train=is_train,
            dataset_path="canonical_dataset.txt",
            sequence_path="canonical_sequence.txt",
        )
        return result
    assert False, "unknown dataset_pipeline_mode {}".format(dataset_pipeline_mode)


def get_dataset_path(is_train):
    """
    Get the path to the dataset.
    """
    dataset_path = "dataset_train_all.h5" if is_train else "dataset_test_0.h5"
    return dataset_path


@permacache(
    "eclip/activate_entire_eclip_sequence_2",
    key_function=dict(filtration_spec=drop_if_equal(dict(type="never_filter"))),
)
def activate_entire_eclip_sequence(
    is_train,
    i,
    *,
    replicate_category,
    dataset_pipeline_mode,
):
    """
    Produce a list of all the positions in the sequence that are contained inside an eclip.

    Parameters
    ----------
    is_train: whether or not to get the training dataset
    i: the data chunk to get
    replicate_category: which replicate category to get
    dataset_pipeline_mode: which dataset pipeline mode to use

    Returns
    -------
    dictionary with keys that represents a sparse array:
        - batch_idxs: the batch index for each position
        - seq_idxs: the sequence index for each position
        - motif_idxs: the motif index for each position
    """
    eclip_data = get_eclip_dataset(
        is_train,
        replicate_category=replicate_category,
        dataset_pipeline_mode=dataset_pipeline_mode,
    )[i]
    with h5py.File(get_dataset_path(is_train), "r") as f:
        ys = f[f"Y{i}"][0]
    ys = to_intron_exon_annotation(ys)

    motif_idxs = []
    batch_idxs = []
    seq_idxs = []
    for motif_idx in range(len(eclip_data)):
        result = np.array(scipy.sparse.find(eclip_data[motif_idx]))
        prev_batch = prev_idx = None
        for batch, seq, indicator in sorted(result.T.tolist()):
            if indicator == 1:
                # start of sequence
                prev_batch, prev_idx = batch, seq
                continue
            # end of sequence
            assert indicator == 2

            def emit(motif_idx, batch_idx, start_idx, stop_idx):
                seq_range = range(start_idx, stop_idx + 1)
                motif_idxs.extend([motif_idx] * len(seq_range))
                batch_idxs.extend([batch_idx] * len(seq_range))
                seq_idxs.extend(seq_range)

            if prev_batch == batch:
                # within a single batch
                emit(motif_idx, batch, prev_idx, seq)
            elif prev_batch is None:
                emit(motif_idx, batch, 0, seq)
            else:
                emit(
                    motif_idx, prev_batch, prev_idx, eclip_data[motif_idx].shape[1] - 1
                )
            prev_batch, prev_idx = None, None
    return dict(
        motif_idxs=np.array(motif_idxs),
        batch_idxs=np.array(batch_idxs),
        seq_idxs=np.array(seq_idxs),
    )


def expand_eclip_original(eclip_data, i, j, x_shape, *, dataset_pipeline_mode):
    """
    Expand out eclip data for the given index, using the original (one-hot) mode.

    Parameters
    ----------
    eclip_data: the parameters for the eclip data
    i: the data chunk to get
    j: the index within the data chunk to get
    x_shape: the (full) shape of the x data, (N, L, 4)
    dataset_pipeline_mode: which dataset pipeline mode to use

    Returns
    -------
    a matrix of shape (L, M, R, 2), where M is the number of motifs, R is the number of
        replicates, and 2 is the one-hot of start vs end
    """
    names = motif_names(eclip_data)
    # (L, M, R, 2)
    result = np.zeros(
        (x_shape[1], len(names), len(eclip_data["replicates_to_use"]), 2), dtype=np.bool
    )
    for replicate_idx, replicate in enumerate(eclip_data["replicates_to_use"]):
        data = get_eclip_dataset(
            eclip_data["is_train"],
            replicate_category=replicate,
            dataset_pipeline_mode=dataset_pipeline_mode,
        )[i]
        assert j is not None, "not supporting getting the entire array as of yet"
        extracted_rows = [x.getrow(j).toarray() for x in data]
        extracted_rows = np.array(extracted_rows)
        # (M, 1, L)
        assert extracted_rows.shape[1] == 1
        extracted_rows = extracted_rows[:, 0, :]
        # (M, L)
        extracted_rows = extracted_rows.T
        # (L, M)
        result[extracted_rows == 1, replicate_idx, 0] = True
        result[extracted_rows == 2, replicate_idx, 1] = True
    return result


def expand_eclip_all_positions_active(
    eclip_data, i, j, x_shape, *, dataset_pipeline_mode, **kwargs
):
    """
    Expand out eclip data for the given index, using the all-positions-active mode.

    If a given site is a peak in multiple replicates, it will be counted multiple times.

    Parameters
    ----------
    eclip_data: the parameters for the eclip data
    i: the data chunk to get
    j: the index within the data chunk to get
    x_shape: the (full) shape of the x data, (N, L, 4)
    dataset_pipeline_mode: which dataset pipeline mode to use

    Returns
    -------
    a matrix of shape (L, M), where M is the number of motifs. Values are in the range
        [0, R], where R is the number of replicates. Each value represents the number of
        eclip peaks of the given motifs that overlap the given position.
    """
    names = motif_names(eclip_data)
    m_out = np.zeros((x_shape[1], len(names)))
    for rc in eclip_data["replicates_to_use"]:
        seq = activate_entire_eclip_sequence(
            eclip_data["is_train"],
            i,
            replicate_category=rc,
            dataset_pipeline_mode=dataset_pipeline_mode,
            **kwargs,
        )

        mask = seq["batch_idxs"] == j
        motif_idxs = seq["motif_idxs"][mask]
        seq_idxs = seq["seq_idxs"][mask]
        np.add.at(m_out, (seq_idxs, motif_idxs), 1)
    return m_out


def motif_names(eclip_data):
    """
    Get the names of the motifs for the given eclip data.

    Checks that the names are consistent across replicates.
    """
    all_names = [
        peak_names(replicate_category=rc) for rc in eclip_data["replicates_to_use"]
    ]
    for names in all_names:
        assert names == all_names[0]
    return all_names[0]


def expand_eclip(eclip_data, i, j, x_shape):
    """
    Convert compressed exclip data to a one-hot motif encoding.

    Returns a matrix of shape (L, M, *), where * depends on the mode.
    """
    return dict(
        all_positions_active=expand_eclip_all_positions_active,
        eclip_original=expand_eclip_original,
    )[eclip_data["one_hot_encoding_mode"]](
        eclip_data,
        i,
        j,
        x_shape,
        dataset_pipeline_mode=eclip_data["dataset_pipeline_mode"],
        **eclip_data.get("one_hot_encoding_kwargs", {}),
    )


class EClipAdditionalData(AdditionalData):
    """
    Wrapper around `expand_eclip` that makes it compatible with the
    `AdditionalData` interface. Runs `expand_eclip` with the given
    parameters as well as the `is_train` parameter, which is determined
    by the path.
    """

    def __init__(self, params):
        self.params = params

    def compute_additional_input(self, original_input, path, i, j):
        is_train = self.classify_path(path)
        return expand_eclip(
            dict(**self.params, is_train=is_train), i, j, original_input[None].shape
        )
