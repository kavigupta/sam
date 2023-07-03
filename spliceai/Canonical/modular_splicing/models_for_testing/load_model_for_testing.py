import os

import numpy as np
import torch

from modular_splicing.utils.io import load_model, model_steps
from modular_splicing.utils.entropy_calculations import hbern

from permacache import permacache


@permacache(
    "spliceai/statistics/model_sparsity_2",
    key_function=dict(path=os.path.realpath, step=int),
)
def model_sparsity(path, step):
    """
    Get the sparsity of a model at a given step.

    Cached since in concept we shouldn't create the same checkpoint twice.

    Parameters
    ----------
    path : str
        Path to the model.
    step : int
        Step of the model.

    Returns
    -------
    sparsity : float
        Sparsity of the model. (Note! Not the density, but the sparsity.)
    """
    print(path, step)
    _, m = load_model(path, step, map_location=torch.device("cpu"))
    return m.sparse_layer.get_sparsity()


def achieved_target_acc(path, pbar=lambda x: x):
    """
    Produce the steps where the model achieves the target accuracy,
        and the sparsity at those steps.

    Loading the model from this step will give the model with the
        target (validation) accuracy and given sparsity.

    Parameters
    ----------
    path : str
        Path to the model.
    pbar : iterable -> iterable (e.g., tqdm.tqdm)
        Progress bar to use.

    Returns
    -------
    steps : array of int
        Steps where the model achieves the target accuracy.
    sparsities : array of float
        Sparsities at those steps.
    """
    steps = np.array(model_steps(path))
    sparsities = np.array([model_sparsity(path, step) for step in pbar(steps)])
    [idxs] = np.where(sparsities[:-1] != sparsities[1:])
    return steps[idxs], sparsities[idxs]


def step_for_density(model_path, target_density, err=True):
    """
    Produce a step for a model where it has achieved the target accuracy
        at the given target density.

    Raises an error if the model has not reached that density, or skipped
        over it (we use a tolerance of 1%). If err is False, then we return
        None instead of raising an error.

    Parameters
    ----------
    model_path : str
        Path to the model.
    target_density : float
        Target density.
    err : bool
        Whether to raise an error if the model has not reached the target
        density.

    Returns
    -------
    step : int
        The step at which the model has achieved the target accuracy at the
        target density.

        None if this function would have raised an error, but err is False.
    """
    if not err and not os.path.exists(model_path):
        return None
    steps, sparsities = achieved_target_acc(model_path)
    densities = 1 - sparsities
    # get differences
    deltas = np.abs(densities - target_density)
    if not deltas.size and not err:
        return None
    idx = deltas.argmin()
    # check if we're within 1% of the target density
    if deltas[idx] / target_density >= 0.01:
        if not err:
            return None
        raise ValueError(
            "Could not find a step for the given density. "
            "The closest step is {} with a density of {}".format(
                steps[idx], densities[idx]
            )
        )
    return int(steps[idx])


def density_for_entropy(
    *,
    target_entropy,
    num_motifs,
    entropy_per_activation,
    starting_density=0.75,
    density_update=0.75,
):
    """
    Compute the maximal density step for a model to get below the target entropy.
        By default, assumes typical starting density of 0.75, and a density
        update of 0.75.

    Parameters
    ----------
    target_entropy : float
        Target entropy, in bits.
    num_motifs : int
        Number of motifs in the model.
    entropy_per_activation : float
        Entropy per activation, in bits.
    starting_density : float
        Starting density of the model.
    density_update : float
        Density update of the model.

    Returns
    -------
    density : float
        The maximal density step for a model to get below the target entropy.
    """
    density = starting_density
    while True:
        entropy = num_motifs * (hbern(density) + density * entropy_per_activation)
        if entropy <= target_entropy:
            return density
        density *= density_update
