from collections import Counter
import functools
import itertools

import tqdm
import torch
import numpy as np

from permacache import permacache, drop_if_equal
from modular_splicing.dataset.basic_dataset import basic_dataset
from modular_splicing.evaluation.run_evaluation import evaluate_model_on_data
from modular_splicing.legacy.hash_model import hash_model


def round_postsparse_values(round, output):
    """
    Rounds the post-sparse values as specified by the `round` parameter.

    If `round < 0`, then we instead set all non-zero values to their mean.

    If `round > 0`, then we set every non-zero value to [round, 2 * round, ...] via a rounding process.
    """
    if round:
        if round < 0:
            for i in range(output.shape[2]):
                output[:, :, i][output[:, :, i] != 0] = output[:, :, i][
                    output[:, :, i] != 0
                ].mean()
        else:
            nonzeros = output != 0
            output = torch.maximum(
                (output / round).round() * round, torch.tensor(round).to(output)
            )
            output = output * nonzeros
    return output


@permacache(
    "postsparse_bits_analysis/accs",
    key_function=dict(m=hash_model, eval_path=drop_if_equal("dataset_test_0.h5")),
)
def rounded_accuracy(m, round, eval_path="dataset_test_0.h5"):
    """
    Evaluate the model on the test set, rounding the post-sparse values as specified by `round`.

    See `round_postsparse_values` for more details.
    """
    deval = basic_dataset(eval_path, m.cl, 10_000, sl=5000)
    return evaluate_model_on_data(
        m,
        deval,
        model_kwargs=dict(
            manipulate_post_sparse=functools.partial(round_postsparse_values, round)
        ),
        pbar=tqdm.tqdm,
        bs=32,
        limit=len(deval) // 2,
    )


@permacache(
    "postsparse_bits_analysis/compute_reasonable_values",
    key_function=dict(m=hash_model, data_path=drop_if_equal("dataset_train_all.h5")),
)
def sample_nonzero_values(m, data_path="dataset_train_all.h5"):
    """
    Samples non-zero values into a numpy array from the model's post-sparse layer.

    Done on the training set.

    Arguments:
        m: The model to sample from.
        data_path: The path to the dataset to sample from.

    Returns:
        A numpy array of the non-zero post-sparse values.
    """
    data = basic_dataset(data_path, m.cl, 10_000, sl=5000)
    xs = []
    post_sparses = []
    for x, _ in itertools.islice(data, 10):
        xs.append(x)
        with torch.no_grad():
            post_sparses.append(
                m(torch.tensor([x]).float().cuda(), collect_intermediates=1)[
                    "post_sparse"
                ]
                .detach()
                .cpu()
                .numpy()
            )
    post_sparse_vals = np.concatenate(post_sparses, axis=0)[:, :, 2:]
    return post_sparse_vals[post_sparse_vals != 0]


def compute_entropy(rounded_values):
    """
    Compute the entropy of an empirical distribution of rounded values.

    Arguments:
        rounded_values: A numpy array of rounded values.

    Returns:
        The entropy of the distribution in bits.
    """
    freqs = np.array(list(Counter(rounded_values).values()), dtype=np.float64)
    freqs /= freqs.sum()
    return -(freqs * np.log(freqs)).sum() / np.log(2)


def entropy(sampled_values, round_amount):
    """
    Computes the entropy of the given sampled values, assuming they are rounded as in
        `round_postsparse_values`.
    """
    if round_amount < 0:
        return 0
    rounded_values = np.maximum(
        (sampled_values / round_amount).round() * round_amount, round_amount
    )
    return compute_entropy(rounded_values)


def accuracy_drop_vs_entropy(m, rounding_values):
    """
    Produce a plot of the accuracy drop vs entropy for the given model and rounding values.
    """
    rv = sample_nonzero_values(m)
    accs_values = [
        np.mean(rounded_accuracy(m, rounding)) - np.mean(rounded_accuracy(m, 0))
        for rounding in rounding_values
    ]
    entropies = [entropy(rv, rounding) for rounding in rounding_values]

    accs_values = np.array(accs_values)
    entropies = np.array(entropies)
    rounding_values = np.array(rounding_values)

    if 0 in entropies:
        ideal = accs_values[entropies == 0]
        mask = accs_values >= ideal
        accs_values = accs_values[mask]
        entropies = entropies[mask]
        rounding_values = rounding_values[mask]

    return dict(
        a=accs_values,
        e=entropies,
        r=rounding_values,
        sparsity=1 - m.get_sparsity(),
    )


def accuracy_drop_vs_entropy_for_model(mod, rounding_values):
    """
    Like `accuracy_drop_vs_entropy`, but for a model.
    """
    return accuracy_drop_vs_entropy(mod.model, rounding_values)
