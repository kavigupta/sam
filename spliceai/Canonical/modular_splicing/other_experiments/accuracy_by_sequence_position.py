from permacache import permacache, stable_hash

import numpy as np
import tqdm.auto as tqdm
import scipy.stats


from modular_splicing.evaluation.predict_splicing import predict_splicepoints

from modular_splicing.data_for_experiments.standardized_sample import (
    standardized_sample,
)


@permacache(
    "modular_splicing/other_experiments/compute_confusion_by_position",
    key_function=dict(model=stable_hash),
)
def compute_confusion_by_position(model, path="dataset_test_0.h5"):
    xs, ys = standardized_sample(path, None, sl=5000, cl=model.cl)
    ys_binary = np.eye(3, dtype=np.bool_)[ys][:, :, 1:]
    yps = predict_splicepoints(model, xs, batch_size=16, pbar=tqdm.tqdm)
    thresholds = [
        np.percentile(yps[:, :, c], 100 * (1 - (ys == (c + 1)).mean()))
        for c in range(2)
    ]
    yps_binary = yps > thresholds
    return confusion_by_position(ys_binary, yps_binary)


def confusion_by_position(ys_true_binary, ys_pred_binary):
    """
    Compute a confusion matrix for each position in the sequence.

    Arguments
    ---------
    ys_true_binary : (N, L, 2)
        The true labels, in binary form.
    ys_pred_binary : (N, L, 2)
        The predicted labels, in binary form.

    Returns
    -------
    confusion : (L, 2, 2)
        The confusion matrix for each position in the sequence.
        confusion[i][true][pred] = (ys_true_binary[:, i, true] & ys_pred_binary[:, i, pred]).sum()
    """
    result = {}
    for c in range(2):
        # (N, L); high bit: true, low bit: pred
        combined = ys_true_binary[:, :, c] * 2 + ys_pred_binary[:, :, c]
        batch_locs, seq_locs = np.where(combined)
        counts = np.zeros((combined.shape[1], 4), dtype=np.int64)
        np.add.at(counts, (seq_locs, combined[batch_locs, seq_locs]), 1)
        counts[:, 0] = combined.shape[0] - counts.sum(1)

        # [true=0, pred=0; true=0, pred=1; true=1, pred=0; true=1, pred=1]
        # ->
        # [[true=0, pred=0; true=0, pred=1], [true=1, pred=0; true=1, pred=1]]

        # counts[seq][true][pred]

        counts = counts.reshape(-1, 2, 2)
        result[c + 1] = counts
    return result


def boostrap_avg_topk(fn1, tp1, fn2, tp2, amount=10_000):
    """
    Boostrap of the average topk accuracy of the two samples, given the
        false negative and true positive counts for each.
    """
    n1 = fn1 + tp1
    n2 = fn2 + tp2
    p1 = tp1 / n1
    p2 = tp2 / n2

    rng = np.random.RandomState(0)

    samples_1 = scipy.stats.binom(n1, p1).rvs(amount, rng) / n1
    samples_2 = scipy.stats.binom(n2, p2).rvs(amount, rng) / n2
    boot = (samples_1 + samples_2) / 2
    return ((p1 + p2) / 2, *np.percentile(boot, [2.5, 97.5]))


def collect_by_chunk(counts, num_chunks=5):
    """
    Collect the counts by chunk, so that we can compute the average topk
        for each chunk.

    Arguments
    ---------
    counts : (L, 2, 2)
        The confusion matrix for each position in the sequence.
    num_chunks : int
        The number of chunks to collect. The last one will be slightly smaller.
        E.g., if L=1000 and num_chunks=3, then the chunks will be 334, 334, 332.

    Returns
    -------
    labels: a label for each chunk
    chunked_counts : (num_chunks, 2, 2)
    """
    chunk_size = (counts.shape[0] + num_chunks - 1) // num_chunks
    sums, labels = [], []
    for i in range(0, counts.shape[0], chunk_size):
        sums.append(counts[i : i + chunk_size].sum(0))
        labels.append(f"{i}-{i + chunk_size - 1}")
    return labels, np.array(sums)


def collect_for_both(result, **kwargs):
    """
    Like collect_by_chunk, but for both classes.
    """
    labels, acc = collect_by_chunk(result[1], **kwargs)
    _, don = collect_by_chunk(result[2], **kwargs)
    return labels, acc, don


def all_results_on_models(models):
    """
    Compute all the results on the given list of ModelForTraining objects.
    """
    results = [compute_confusion_by_position(mod.model) for mod in tqdm.tqdm(models)]
    results = [collect_for_both(result, num_chunks=2) for result in results]
    return results


def aggregate_confusion_results(results):
    """
    Aggregates results of `collect_for_both` for multiple models.

    Effectively, just adds together the counts.
    """
    labelses, acces, dones = zip(*results)
    assert all(label == labelses[0] for label in labelses)
    labels = labelses[0]
    acc = sum(acces)
    don = sum(dones)
    return labels, acc, don


def draw_result(ax, labels, acc, don, *, label, off, **kwargs):
    """
    Draws the results of `collect_for_both` on the given axis.
    """
    xs = np.arange(len(labels)) + off
    boots = [boostrap_avg_topk(*a[1], *d[1]) for a, d in zip(acc, don)]
    mu, lo, hi = [100 * np.array(x) for x in zip(*boots)]
    ax.scatter(xs, mu, **kwargs, label=label)
    ax.errorbar(xs, (lo + hi) / 2, (hi - lo) / 2, **kwargs, capsize=3, linestyle=" ")
    ax.set_xticks(xs, labels)


def draw_results(results, ax):
    """
    Draws the results of `collect_for_both` on several models, on the given axis.

    Draws each one individually, then draws the aggregate.
    """
    for i, result in enumerate(results):
        draw_result(
            ax,
            *result,
            color="purple",
            label="Individual" if i == 0 else None,
            off=i / (len(results) - 1) * 0.5 - 0.25,
            alpha=0.5,
        )
    draw_result(
        ax, *aggregate_confusion_results(results), color="black", label="Mean", off=0
    )
    ax.set_xlabel("Position in sequence")
    ax.set_ylabel(f"Top-k Calibrated TPR [%]")
    ax.grid(axis="y")
    ax.legend()
