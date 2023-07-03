"""
Contains and a class for converting a set of motif-specific matching model to
something that's compatible with functions expecting an end-to-end model
"""

from permacache import permacache, stable_hash, drop_if_equal

import numpy as np
import torch
import torch.nn as nn
import tqdm.auto as tqdm

from modular_splicing.data_for_experiments.standardized_sample import (
    standardized_sample,
)
from modular_splicing.fit_rbns.chenxi_neural_model import run_rbns_chenxi_model
from modular_splicing.utils.run_batched import run_batched

# TODO undo the negative convention, within this file. First check that everything works.


def individual_model_function_for_type(mod, model_type):
    """
    Run the given model on the given sequence, and return the result.

    More negative numbers indicate a higher probability of binding, which is
    the opposite of the normal convention.

    Parameters
    ----------
    mod: a model.
    model_type: either "eclip", "rbns", or "rbns_chenxi"

    Returns
    -------
    A function from (N, L, 4) to (N, L)
    """
    return dict(
        rbns_chenxi_fixed=lambda x: -run_rbns_chenxi_model(mod, x),
        rbns_using_am=lambda x: -mod(x)["motifs"][:, :, 0],
        eclip=mod,
    )[model_type]


@permacache(
    "eclip/eclip_model/calibrate_on_genome",
    key_function=dict(mod=stable_hash, bs=None, model_type=drop_if_equal("eclip")),
)
def thresholds_on_genome_for_sparsity(
    mod, *, sparsity, bs=100, cl=50, amount=20_000, model_type="eclip"
):
    """
    Compute the thresholds for the given model on the given sparsity. Note:
        these thresholds are negative. They correspond to the thresholds
        for running the model in a `negative` mode.

        See `indidivuda_model_function_for_type` for more details.

    Parameters
    ----------
    mod: the model to compute thresholds for.
    sparsity: the sparsity to compute thresholds for.
    bs: the batch size to use.
    cl: the chunk length to use.
    amount: the number of sequences to pull from the dataset
    model_type: see individual_model_function_for_type for details.

    Returns
    -------
    A list of thresholds, one for each motif. Each threshold, on the given sequences
    is the threshold that gives the given sparsity.
    """
    xs, _ = standardized_sample("dataset_train_all.h5", amount, cl=cl)

    res = run_batched(
        individual_model_function_for_type(mod, model_type), xs, bs, pbar=tqdm.tqdm
    )

    res = res[:, cl // 2 : res.shape[1] - cl // 2]
    # these are the thresholds that give the given sparsity
    # note that these are negative
    # if they were positive, we would do np.quantile(res, 1 - sparsity)
    return np.quantile(res, sparsity)


class EclipMatchingModelMotifModelStub(nn.Module):
    """
    Represents a stub model that looks like a full end-to-end model
    but only uses motifs. It takes in the given `models` and uses them
    to compute the motifs when asked. By default, it uses the threshold
    0, but if `sparsity` is not None, it computes the thresholds on the
    genome for the given sparsity.
    """

    def __init__(self, cl, models, sparsity, model_type="eclip", pbar=lambda x: x):
        super().__init__()
        if sparsity is not None:
            sparsities = np.array(
                [
                    thresholds_on_genome_for_sparsity(
                        model, sparsity=sparsity, model_type=model_type
                    )
                    for model in pbar(models)
                ]
            )
            self.thresholds = nn.Parameter(torch.tensor(sparsities))
        else:
            self.thresholds = None
        self.models = nn.ModuleList(models).cpu()
        self.cl = cl
        self._use_stable_hash_directly = True
        self.model_type = model_type

    def forward(self, x, only_motifs):
        assert only_motifs is True
        model_type = getattr(self, "model_type", "eclip")
        # get the motifs
        x = [
            individual_model_function_for_type(model, model_type)(x)
            for model in self.models
        ]
        x = torch.stack(x)
        x = x.transpose(0, 1)
        if self.thresholds is not None:
            # if we have thresholds, use them. e.g., -5 is a threshold
            # we subtract that from -7, to get -2, which is good
            # because we're using the negative convention, large values need to be negated.
            x = x - torch.tensor(self.thresholds, device=x.device)[:, None]
            # so now we filter out, e.g., -3 which got sent to +2.
            x[x > 0] = 0
        x = x.transpose(1, 2)
        x = -x
        return dict(post_sparse_motifs_only=x > 0)
