import itertools
from collections import defaultdict

import numpy as np
import tqdm.auto as tqdm
from more_itertools import chunked
from permacache import drop_if_equal, permacache

from modular_splicing.dataset.h5_dataset import H5Dataset
from modular_splicing.eclip.data.eclip_peaks import eclips_from_onehot
from modular_splicing.eclip.data.eclips_for_testing import (
    create_control_eclips,
    extract_actual_range,
)
from modular_splicing.eclip.test_motifs.names import peak_names
from modular_splicing.utils.arrays import add_cl_in
from modular_splicing.utils.intron_exon_annotation import (
    ANNOTATION_EXON,
    ANNOTATION_INTRON,
)


class EclipMotifsDataset:
    """
    Dataset for training eclip models.

    Contains data for each motif, for each of the 4 possible combinations of
        correct/incorrect and exon/intron.

    The data is in the form of sequences. For this to work, you must use a
        mode that leads to constant-length sequences, such as "from_5'".
    """

    def __init__(
        self,
        *,
        path,
        path_annotation,
        seed,
        amount=None,
        batch_size_for_construction=100,
        mode,
    ):
        self.edata = eclip_pattern_dataset(
            path=path,
            path_annotation=path_annotation,
            eclip_params=dict(
                replicates_to_use=["1", "2"],
                one_hot_encoding_mode="eclip_original",
                dataset_pipeline_mode="with_spliceai_pipeline",
            ),
            amount=amount,
            batch_size=batch_size_for_construction,
            mode=mode,
        )
        self.eclip_names = peak_names(replicate_category="1")
        self.seed = seed

    def batched_data(self, motif, batch_size, pbar=lambda x, **_: x):
        """
        Get data in batches.

        Parameters:
            motif: The motif to use
            batch_size: The batch size
            pbar: A function that takes an iterable and returns an iterable.
                This is used to wrap the data in a progress bar.

        Returns: a generator over batches (x, y) of data
            where
                x is a numpy array of shape (batch_size, common_length)
                    with values in {0, 1, 2, 3}
                y is a numpy array of shape (batch_size)
                    with value 0 meaning incorrect and 1 meaning correct

            This dataset is guaranteed to be shuffled and also balanced
                across the 4 categories.
        """
        assert batch_size % 4 == 0
        batch_size //= 4

        edata = self.edata[self.eclip_names.index(motif)]

        segments = [
            (correct, area)
            for correct in [True, False]
            for area in [ANNOTATION_EXON, ANNOTATION_INTRON]
        ]
        edata = {s: np.array(edata[s]) for s in segments}

        length = min(len(edata[s]) for s in segments)

        indices = {s: np.arange(len(edata[s])) for s in segments}

        rng = np.random.RandomState(self.seed)
        for s in segments:
            rng.shuffle(indices[s])

        for start in pbar(range(0, length, batch_size)):
            x_for_each = {
                s: edata[s][indices[s][start : start + batch_size]] for s in segments
            }
            x_overall, y_overall = [], []
            for s in segments:
                x_overall.append(x_for_each[s])
                y_overall.append([s[0]] * len(x_for_each[s]))
            x_overall, y_overall = np.concatenate(x_overall), np.concatenate(y_overall)
            yield x_overall, y_overall


@permacache(
    "eclip/eclip_pattern_dataset_6",
    key_function=dict(
        cl_max=drop_if_equal(10_0000),
        pad_left=drop_if_equal(10),
        pad_right=drop_if_equal(20),
    ),
)
def eclip_pattern_dataset(
    *, path, path_annotation, eclip_params, amount, cl_max=10_000, mode, batch_size
):
    """
    Returns a dataset of eclip motif patterns.

    See `extract_eclip_motif_patterns` for documentation on the parameters.

    Returns: dict[motif_idx -> dict[(outcome, category) -> list[pattern]]]
        where outcome is either True (real eclip) or False (control eclip),
            category is either ANNOTATION_EXON or ANNOTATION_INTRON,
            and pattern is a numpy array of shape (L2, 4),
            where L2 depends on mode and the eclip
    """
    patterns = extract_eclip_motif_patterns(
        path=path,
        path_annotation=path_annotation,
        eclip_params=eclip_params,
        amount=amount,
        cl_max=cl_max,
        mode=mode,
        batch_size=batch_size,
    )
    result = defaultdict(lambda: defaultdict(list))
    for outcome, category, motif_idx, x in patterns:
        result[motif_idx][outcome, category].append(x.argmax(-1))
    return {k: dict(v.items()) for k, v in result.items()}


def extract_eclip_motif_patterns(
    *, path, path_annotation, eclip_params, amount, cl_max, mode, batch_size
):
    """
    Extract all patterns from eclip data.

    Parameters
    ----------
    path : str
        Path to eclip data.
    path_annotation : str
        Path to intron/exon annotation data.
    eclip_params : dict
        Parameters for loading eclip data.
    amount : int
        Number of batches to use.
    cl_max : int
        Maximum context length in the underlying dataset
    mode : str
        the mode to use for extracting the patterns. See `extract_actual_range` for
        details.
    batch_size : int
        Batch size for loading the data.

    Returns the same as `extract_eclip_motif_patterns_for_chunk`, just over the entire dataset.
    """
    standard_kwargs = dict(cl=cl_max, sl=5000, cl_max=cl_max)

    dset_eclip = H5Dataset(
        path=path,
        **standard_kwargs,
        datapoint_extractor_spec=dict(
            type="BasicDatapointExtractor",
            rewriters=[dict(type="eclip", params=eclip_params)],
        ),
        post_processor_spec=dict(
            type="FlattenerPostProcessor",
            indices=[("inputs", "x"), ("inputs", "motifs")],
        ),
    )
    dset_annotation = H5Dataset(
        path=path_annotation,
        **standard_kwargs,
        datapoint_extractor_spec=dict(type="BasicDatapointExtractor", run_argmax=False),
        post_processor_spec=dict(
            type="FlattenerPostProcessor", indices=[("outputs", "y")]
        ),
    )
    if amount is None:
        amount = len(dset_eclip)

    dset = zip(dset_eclip, dset_annotation)

    for batch in chunked(
        tqdm.tqdm(itertools.islice(dset, amount), total=amount), batch_size
    ):
        batch_eclip, batch_annotation = zip(*batch)
        x, m = zip(*batch_eclip)
        (annotations,) = zip(*batch_annotation)
        yield from extract_eclip_motif_patterns_for_chunk(
            x=x, m=m, annotation=annotations, cl=cl_max, mode=mode
        )


def extract_eclip_motif_patterns_for_chunk(*, x, m, annotation, cl, mode):
    """
    Extract eclip motif patterns from a chunk of data.

    Parameters:
    x: (N, L + cl, 4)
        the RNA sequences
    m: (N, L + cl, M, R, 2)
        the eclip motifs in one-hot format
    annotation: (N, L + cl)
        the annotation of the sequences, as either intron or exon
    cl: int
        the context length
    mode:
        the mode to use for extracting the patterns. See `extract_actual_range` for
        details.

    Returns: a sequence of (outcome, category, motif_idx, pattern) tuples, where
        outcome is either True (real eclip) or False (control eclip)
        category is either ANNOTATION_EXON or ANNOTATION_INTRON
        motif_idx is the index of the motif
        pattern is a (L2, 4) array of the pattern
            L2 can theoretically vary from run to run, depending on mode
    """
    m = np.array([m_[cl // 2 : -cl // 2] for m_ in m])
    x = np.array(x)
    annotation = np.array(annotation)
    annotation = annotation[:, :, 0]
    annotation = add_cl_in(annotation, x.shape[1] - annotation.shape[1], pad_value=-100)
    eclips = eclips_from_onehot(m)
    b, l, *_ = m.shape

    control_eclips = create_control_eclips(eclips, np.arange(b), l)

    by_outcome = {True: eclips, False: control_eclips}
    for outcome, elements in by_outcome.items():
        for eclip in elements:
            start, end = extract_actual_range(eclip, mode)
            batch = eclip["batch_idx"]

            slic = batch, slice(cl // 2 + start, cl // 2 + end)
            category_r = annotation[slic]
            category = category_r[0]
            if not (category == category_r).all():
                continue
            yield outcome, category, eclip["motif_idx"], x[slic]
