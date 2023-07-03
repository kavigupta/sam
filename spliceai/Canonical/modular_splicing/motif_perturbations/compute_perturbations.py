"""
Variety of functions for computing motif perturbations.
"""

from types import SimpleNamespace
import numpy as np
import torch

from permacache import permacache, stable_hash
from modular_splicing.evaluation.predict_splicing import predict_splicepoints

from modular_splicing.utils.run_batched import run_batched


def motif_perturbations_generic(
    m,
    x,
    y,
    feature_inclusion_threshold,
    bs,
    pbar,
    get_perturbations,
    threshold_info,
    num_output_channels=3,
):
    """
    Run perturbations on a model and return the resulting outputs. The exact perturbations
        to perform are specified by the get_perturbations callback function.

    Throughout, we use P1 to refer to the number of motifs that we consider perturbing, and P2
        to the number of perturbations that are actually performed.

    F is the number of relevant features.

    Args:
        m: The model to run perturbations on.
        x: The input sequence to run perturbations on: (L, 4)
        y: The expected output of the model: (L, num_output_channels)
        feature_inclusion_threshold: the threshold for which false positive features should be tracked.
            Setting this to a lower value increases the number of results returned, but
            you can always filter out the results later as this function does not perform aggregation.
            This is just to save space, as the number of perturbations is not affected.
        bs: The batch size to use when running the model. Should not affect the results.
        pbar: A progress bar to use when running the model. Should not affect the results.
        get_perturbations: A callback function that
            takes in arguments
                - indices: list of length num_unique_motif_ids. This is basically just a mapping from
                    the ids we are using to the indices of the motifs in the motif matrix.
                    Often, but not always, this is identity.
                - motif_ids (P1,): the list of all motif ids.
                - motif_pos (P1,): the list of all motif positions.
                - motif_vals (P1,): the list of all motif values pre-perturbation.
                - post_sparse (P1, L, M + 2): the list of all motif values post-perturbation.
                    The first two channels are the splicepoints, which should be unchanged
            and outputs
                - motif_vals_kwargs: a dictionary of extra arguments to pass to the output.
                - perturbed_motifs: (P2, L, M + 2) the list of all motif values post-perturbation
        threshold_info: an output of `motif_statistics`.

    Returns a SimpleNamespace with attrs:
        - motif_pos (P1,): array of motif positions that are perturbed
        - motif_ids (P1,): array of motif ids that are perturbed
        - motif_vals (P1,): array of motif values that are perturbed
        - pred (F,): array of predicted values for the relevant features (sans perturbation)
        - actual (F,): array of actual values for the relevant features
        - relevant_idxs (2, F): array of indices of the relevant features
        - perturbed_pred: (P2, F) array of predicted values for the relevant features (with perturbation)
        - whatever extras are returned by get_perturbations
    """
    # run the model to get its unperturbed output
    yp = predict_splicepoints(m, np.array([x]), bs)[0]
    assert yp.shape[-1] == num_output_channels - 1, str(yp.shape)

    # get the motifs from the model
    with torch.no_grad():
        m.eval()
        res = m(torch.tensor([x]).cuda(), only_motifs=True)
        res = {k: v.cpu().numpy() for k, v in res.items()}
    # get the indices of the relevant motifs from the model. Not offset by the splicepoint motifs,
    # so by default it's 0-79 (yes, including the extra padding motif at the end)
    indices = sorted(m.sparsity_enforcer.motif_index())
    # get the motif mask, the mask of the motifs that will be perturbed
    if threshold_info is not None:
        motif_mask = (
            res["nonsparse_motifs"][0][:, indices] > threshold_info.near_miss_thresholds
        )
    else:
        motif_mask = res["post_sparse_motifs_only"][0][:, indices] > 0
    # get the positions, ids, and values of the motifs that will be perturbed
    motif_pos, motif_ids = np.where(motif_mask)
    motif_vals = res["post_sparse_motifs_only"][
        0, motif_pos, np.array(indices)[motif_ids]
    ]
    post_sparse = res["post_sparse"]
    # run the get_perturbations callback function to get the perturbations, along with extra info
    # that will be returned
    motif_vals_kwargs, perturbed_motifs = get_perturbations(
        indices, motif_ids, motif_pos, motif_vals, post_sparse
    )
    splicepoint_results_residual = torch.tensor(
        res["splicepoint_results_residual"]
    ).cuda()

    # compute the relevant features. These are the ones that will be tracked
    actual, pred, relevant_idxs = relevant_features(
        yp,
        y,
        include_threshold=feature_inclusion_threshold,
        num_output_channels=num_output_channels,
    )

    def run_rest_of_model(ms):
        """
        Runs the rest of the model. Only return the relevant features.
        Takes an object of shape (N, L, M + 2) and returns an object of shape (N, F)
        """
        with torch.no_grad():
            m.eval()
            n = ms.shape[0]
            output = m.forward_post_motifs(
                post_sparse=ms,
                splicepoint_results_residual=splicepoint_results_residual.repeat(
                    n, 1, 1
                ),
                collect_intermediates=False,
                collect_losses=False,
            )["output"].softmax(-1)[:, :, 1:]
        return output[:, relevant_idxs[0], relevant_idxs[1]]

    if perturbed_motifs.size > 0:
        outputs = run_batched(run_rest_of_model, perturbed_motifs, bs, pbar=pbar)
    else:
        outputs = []
    result = dict(
        motif_pos=motif_pos,
        motif_ids=motif_ids,
        motif_vals=motif_vals,
        pred=pred,
        actual=actual,
        relevant_idxs=relevant_idxs,
        perturbed_pred=outputs,
    )
    result.update(motif_vals_kwargs)
    return SimpleNamespace(**result)


def relevant_features(yps, ys, include_threshold, num_output_channels):
    """
    Produce the relevant features for a given model output and expected output.

    Effectively, return all true positive features and false positive features with
    a score above the include_threshold.

    Args:
        yps: The model output. Shape (N, L, num_output_channels).
        ys: The expected output. Shape (N, L). In integer rather than 1-hot format.
        include_threshold: The threshold for which false positive features should be tracked.
        num_output_channels: The number of output channels of the model. Zeroeth channel
            is the null prediction, so e.g., if you are predicting A or D it should be 3.

    Returns:
        actual: the actual values. Shape (F,) where F is the number of relevant features.
        pred: the predicted values. Shape (F,)
        relevant_idxs: tuple of indices of the relevant features. Shape (2, F) where the first row
            is the indices of the relevant positions and the second row is the indices of the
            relevant channels (not including the null prediction)
    """
    one_hot_ys = np.eye(num_output_channels, dtype=np.bool)[ys][:, 1:]
    relevant_idxs = np.where((yps > include_threshold) | (one_hot_ys != 0))
    pred = yps[relevant_idxs]
    actual = one_hot_ys[relevant_idxs]
    return actual, pred, relevant_idxs


@permacache(
    "modular_splicing/motif_perturbations/compute_perturbations/motif_perturbations_individual",
    key_function=dict(
        m=stable_hash,
        x=stable_hash,
        y=stable_hash,
        threshold_info=stable_hash,
        bs=None,
        pbar=None,
    ),
)
def motif_perturbations_individual(
    m,
    x,
    y,
    threshold_info,
    *,
    include_threshold=0.001,
    pbar=lambda x: x,
    bs=32,
    num_output_channels=3,
):
    """
    See `motif_perturbations_generic` for more details.

    The idea here is to perturb individual motifs. We thus have that P1 = P2

    We return the extra key `perturbed_motif_vals`, that matches the size of `motif_vals`. It
        contains the values of the motifs after perturbation. The values are in the same order
        as the motifs in `motif_vals`.
    """

    def perturb_individually(indices, motif_ids, motif_pos, motif_vals, post_sparse):
        n_perturb = motif_pos.size
        new_vals = []
        for k in range(n_perturb):
            if motif_vals[k] == 0:
                if threshold_info is not None:
                    new_val = threshold_info.mean_above_thresh[motif_ids[k]]
                else:
                    new_val = 1
                new_vals.append(new_val)
            else:
                new_vals.append(0)
        perturbed_motifs = np.repeat(post_sparse, n_perturb, axis=0)
        # offset of 2 is necessary since the perturbed motifs includes
        # the splice sites.
        perturbed_motifs[
            np.arange(n_perturb), motif_pos, np.array(indices)[motif_ids] + 2
        ] = new_vals
        motif_vals_kwargs = dict(perturbed_motif_vals=np.array(new_vals))
        return motif_vals_kwargs, perturbed_motifs

    return motif_perturbations_generic(
        m,
        x,
        y,
        include_threshold,
        bs,
        pbar,
        get_perturbations=perturb_individually,
        threshold_info=threshold_info,
        num_output_channels=num_output_channels,
    )
