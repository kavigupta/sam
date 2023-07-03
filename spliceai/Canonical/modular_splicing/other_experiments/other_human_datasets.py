import tqdm.auto as tqdm
import numpy as np

from permacache import permacache, stable_hash
from modular_splicing.dataset.alternative.merge_dataset import merged_dataset

from modular_splicing.utils.run_batched import run_batched


@permacache(
    "modular_splicing/other_experiments/other_human_datasets/run_model_on_all_2",
    key_function=dict(
        m=stable_hash, canonical_data=stable_hash, splice_tables=stable_hash
    ),
)
def run_model_on_all(
    m, canonical_data, splice_tables, *, ref_genome_path, l_max, CL_max=10_000
):
    """
    Run model on all the given datasets.

    Parameters
    ----------
    canonical_data, splice_tables: first two outputs of `splice_tables_on_common_genes`
    ref_genome_path: path to the reference genome fa file
    l_max: maximum L value that will later be used when calculating top-kL accuracy. You can put a value
        that's much larger than the maximum you will later use, but not smaller. The smaller the l_max,
        the less values are returned, which saves disk space, loading time, and computation of
        top-kl accuracy down the line.
    CL_max: 10k. Put as a parameter just for documentation reasons, but `merged_dataset`
        has it hardcoded to 10k.

    Returns
    -------
    dictionary mapping channel (1, 2) to a dictionary with keys
        pred: predicted scores at each site (in range [0, 1])
        true: dictionary mapping data_key to array of bool
        count_all: equivalent to np.any(true.values(), axis=0).sum()
    """
    x, ys_each = merged_dataset(
        canonical_data=canonical_data,
        data_txt_files=splice_tables,
        files_for_gene_intersection=["can"],
        data_segment_chunks_to_use=[("test", "0")],
        ref_genome_path=ref_genome_path,
    )
    x, ys_each = x["test", "0"], ys_each["test", "0"]

    indices = range(len(x))

    x = np.concatenate([x[f"X{i}"] for i in indices])

    trim = (CL_max - m.cl) // 2
    x = x[:, trim : x.shape[1] - trim]

    ys_each = {
        k: np.concatenate([ys_each[k][f"Y{i}"].original for i in indices])
        for k in ys_each
    }
    ys_any = 0
    for k in ys_each:
        ys_any |= ys_each[k]
    assert ys_any.max() == 2
    k_spls = [(ys_any == c).mean() for c in (1, 2)]
    yps = run_batched(
        lambda x: m(x).softmax(-1), np.eye(4, dtype=np.float32)[x], 16, pbar=tqdm.tqdm
    )
    results_by_channel = {}
    for c in (1, 2):
        for_c = yps[:, :, c]
        thresh = np.quantile(for_c, 1 - l_max * k_spls[c - 1])
        mask = (for_c >= thresh) | (ys_any == c)
        results_by_channel[c] = dict(
            pred=for_c[mask],
            true={k: (ys_each[k] == c)[mask] for k in ys_each},
            count_all=(ys_any == c).sum(),
        )
    return results_by_channel
