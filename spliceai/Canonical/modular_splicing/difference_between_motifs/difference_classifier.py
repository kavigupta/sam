from collections import Counter
import itertools

from permacache import permacache, stable_hash
import tqdm.auto as tqdm

import numpy as np
from sklearn.linear_model import LogisticRegression

from modular_splicing.data_for_experiments.standardized_sample import (
    standardized_sample,
)
from modular_splicing.evaluation.predict_motifs import predict_motifs_binarized_sparse
from modular_splicing.utils.sequence_utils import decode_packed_seq, pack_sequences


def gather_matching_windows_for_motif_mask(xs, mot, *, cl, w):
    """
    Return the sites that match the given motif, with w bases total around
    the center.

    Parameters
    ----------
    xs : (N, L + CL, 4)
        The underlying sequences to sample from.
    mot : (N, L + CL)
        Mask of where the motif is.
    cl : int
        The context length
    w : int
        The window size of the motif windows to return

    Returns
    -------
    (N, W, 4)
        The windows whose center matches the motif.
    """
    # clip of the context bases
    mot = mot[:, cl // 2 : -cl // 2]
    # Find the sites that match the motifs
    positives = np.array(np.where(mot)).T
    # pull out the relevant sequences and reshape them to be N, L, C
    batch_i, seq_i = positives.T
    seq_i, off_i = np.meshgrid(seq_i, np.arange(-(w // 2), 1 + (w // 2)))
    seq_i = seq_i + off_i + cl // 2
    positives = xs[batch_i, seq_i].transpose((1, 0, 2))
    # Filter sequences for those that have no Ns
    positives = positives[(positives.sum(-1) == 1).all(-1)]
    return positives


@permacache(
    "modular_splicing/difference_between_motifs/difference_classifier/matches_for_motifs",
    key_function=dict(model=stable_hash, xs=stable_hash),
)
def matches_for_motifs(model, xs, w):
    """
    Find the matches for a each motif in the given sequences, for a given model.

    Parameters
    ----------
    model : an end to end model
        The model to use to find the matches.
    xs : (N, L, 4)
        The sequences to find matches for.
    w : int
        The window size of the motif windows to return
    Returns
    -------
    a list of length M, where M is the number of motifs in the model.
    Each element is a numpy array containing packed sequences of the matching patterns.
    """
    mots = predict_motifs_binarized_sparse(model, xs).original[:, :, :79]

    matches = [
        gather_matching_windows_for_motif_mask(xs, mots[:, :, i], cl=model.cl, w=w)
        for i in tqdm.trange(mots.shape[-1])
    ]
    matches = [pack_sequences(x.argmax(-1)) for x in matches]

    return matches


def bag_difference(a, b):
    """
    Take the difference of two bags of sequences, returning the sequences that are in a but not in b.

    Duplicates are treated as distinct, with counts being subtracted.

    E.g., bag_difference([1, 1, 2], [1, 2]) -> [1]
    """
    a = Counter(a)
    b = Counter(b)
    counts = {k: a[k] - b[k] for k in a if a[k] > b[k]}
    if len(counts) == 0:
        return np.array([], dtype=np.int64)
    flattened = np.concatenate([[k] * counts[k] for k in counts])
    return flattened


@permacache(
    "modular_splicing/difference_between_motifs/difference_classifier//difference_of_motifs",
    key_function=dict(fm=stable_hash, am=stable_hash),
)
def difference_of_motifs(fm, am, w):
    """
    Train a model to distinguish between the given two motifs.

    Positive examples are the sequences that are in am but not in fm.
    Negative examples are the sequences that are in fm but not in am.

    Parameters
    ----------
    fm : (N_1,)
        The sequences that are in the foreground motif, packed.
    am : (N_2,)
        The sequences that are in the background motif, packed.
    w : int
        The window size of the motif windows to return

    Returns
    -------
    A dictionary containing the following keys:
    - discriminator_model: the model that was trained
    - fm_only_psam: the psam of the sequences that are in fm but not in am
    - am_only_psam: the psam of the sequences that are in am but not in fm
    - fm_psam: the psam of the sequences that are in fm
    - am_psam: the psam of the sequences that are in am
    - train_accuracy: the accuracy of the model on the training data
    """

    fm_psam, am_psam = [np.eye(4)[decode_packed_seq(k, w)].mean(0) for k in (fm, am)]

    fm_only, am_only, x_val, y_lab = set_up_difference_data(fm, am, w)

    fm_only_psam, am_only_psam = fm_only.mean(0), am_only.mean(0)

    if len(am_only) > 0 and len(fm_only) > 0:
        model = LogisticRegression(random_state=0, solver="liblinear").fit(
            x_val.reshape(x_val.shape[0], -1), y_lab
        )
        train_accuracy = (
            model.predict(x_val.reshape(x_val.shape[0], -1)) == y_lab
        ).mean()

    else:
        model = None
        train_accuracy = 1

    return dict(
        fm_only_psam=fm_only_psam,
        am_only_psam=am_only_psam,
        fm_psam=fm_psam,
        am_psam=am_psam,
        discriminator_model=model,
        train_accuracy=train_accuracy,
    )


def train_all_differences_of_motifs(fms, ams, fms_tests, ams_tests, w):
    """
    Train models to distinguish betwen the given sets of motifs.
    Add the key `eval_accuracies` to each model,
        which is a list of all accuracies on the various test sets

    Parameters
    ----------
    fms : list[M] of 1d arrays, corresponding to the FM motifs
    ams : list[M] of 1d arrays, corresponding to the AM motifs
    fms_tests : list[K][M] of 1d arrays, where K is the number of test sets and M is the number of motifs
    ams_tests : list[K][M] of 1d arrays, where K is the number of test sets and M is the number of motifs
    Returns:
    -------
    A list[M] of dictionaries, where each dictionary is the output of difference_of_motifs
        with the key `eval_accuracies` added.
    """
    fms_tests, ams_tests = list(zip(*fms_tests)), list(zip(*ams_tests))
    results = []
    for fm, am, fm_tests, am_tests in tqdm.tqdm(
        list(zip(fms, ams, fms_tests, ams_tests)), desc="Training models"
    ):
        result = difference_of_motifs(fm, am, w).copy()
        result["eval_accuracies"] = [
            evaluate_motif(result["discriminator_model"], fm_test, am_test, w)
            for fm_test, am_test in zip(fm_tests, am_tests)
        ]
        results.append(result)
    return results


def set_up_difference_data(fm, am, w):
    """
    Set up the data for training a model to distinguish between the given two motifs.

    Positive examples are the sequences that are in am but not in fm.
    Negative examples are the sequences that are in fm but not in am.

    Parameters
    ----------
    fm, am: arrays of packed sequences

    Returns
    -------
    fm_only, am_only, x_val, y_lab

    fm_only: (N, W, 4) the sequences that are in fm but not in am
    """
    fm_only = bag_difference(fm, am)
    am_only = bag_difference(am, fm)

    fm_only, am_only = [np.eye(4)[decode_packed_seq(k, w)] for k in (fm_only, am_only)]

    x_val = np.concatenate([fm_only, am_only])
    y_lab = np.concatenate([np.zeros(fm_only.shape[0]), np.ones(am_only.shape[0])])

    return fm_only, am_only, x_val, y_lab


def evaluate_motif(model, fm, am, w):
    """
    Evaluate the given model on the given test set.
    """
    if model is None:
        return 1
    _, _, x, y = set_up_difference_data(fm, am, w)
    return (model.predict(x.reshape(x.shape[0], -1)) == y).mean()


def render_difference_model(name, o, w):
    """
    Render the given difference model to an image.
    """
    results = [
        o["fm_only_psam"],
        o["am_only_psam"],
        o["discriminator_model"].coef_.reshape(w, 4)
        if o["discriminator_model"] is not None
        else np.zeros((w, 4)) + np.nan,
    ]

    labels = [
        rf"{name} [FM \ AM]",
        rf"{name} [AM \ FM]",
        rf"{name} [dsc ta={o['train_accuracy']:.2%}, va={o['eval_accuracy']:.2%}]",
    ]
    return results, labels


@permacache(
    "modular_splicing/difference_between_motifs/difference_classifier/full_psam_difference_analysis_3",
    key_function=dict(models=stable_hash),
)
def full_psam_difference_analysis(models, fm_key, am_keys, *, w):
    """
    Run a full analysis of the difference between the given models.
    """
    xs_train, _ = standardized_sample(
        "dataset_train_all.h5", 10_000, cl=models[fm_key].cl
    )
    xs_test, _ = standardized_sample("dataset_test_0.h5", 1000, cl=models[fm_key].cl)
    matches = {
        k: matches_for_motifs(models[k], xs_train, w=w)
        for k in tqdm.tqdm(models, desc="collect matches train")
    }
    matches_test = {
        k: matches_for_motifs(models[k], xs_test, w=w)
        for k in tqdm.tqdm(models, desc="collect matches test")
    }
    results = {}
    for k in am_keys:
        results[k] = train_all_differences_of_motifs(
            matches[fm_key],
            matches[k],
            [matches_test[fm_key] for _ in am_keys],
            [matches_test[kprime] for kprime in am_keys],
            w,
        )
        for mot_res in results[k]:
            mot_res["eval_accuracies"] = dict(zip(am_keys, mot_res["eval_accuracies"]))
            mot_res["eval_accuracy"] = mot_res["eval_accuracies"][k]
    return results, matches_test, xs_test.shape[0] * xs_test.shape[1]


def bag_to_set(xs):
    """
    Convert a bag of sequences to a set of sequences. Uses the projection
        x -> (x, index)
    where index is the index of x in the multiplicity.

    E.g., [1, 2, 2, 1, 3, 3] -> {(1, 0), (2, 0), (2, 1), (1, 1), (3, 0), (3, 1)}
    """
    xs = Counter(xs)
    return {(x, i) for x, m in xs.items() for i in range(m)}


def venn3_intersect_bag_sizes(xs, ys, zs):
    """
    Compute the sizes of the intersections of the given bags. The
        result is a dictionary form strings "100", "010", "001", "110", etc.
        to integers, counting the number of elements in the intersection.

    Parameters
    ----------
    xs, ys, zs: bags of sequences

    Returns
    -------
    A dictionary from strings to integers, counting the number of elements
        in the intersection.
    """
    sets = [bag_to_set(x) for x in (xs, ys, zs)]
    universal = list(set.union(*sets))
    bitvectors = np.array([[x in s for x in universal] for s in sets])
    keys = itertools.product(*[[0, 1] for _ in range(3)])
    counts = {
        "".join(map(str, k)): (bitvectors == np.array(k)[:, None]).all(0).sum()
        for k in keys
    }
    return counts
