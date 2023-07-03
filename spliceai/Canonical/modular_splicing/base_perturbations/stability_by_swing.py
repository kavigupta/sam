from permacache import permacache, stable_hash
import tqdm.auto as tqdm

import numpy as np
from modular_splicing.base_perturbations.perturbations import (
    probabilities_by_changed_sequence,
)

from modular_splicing.utils.run_batched import run_batched


@permacache(
    "perturbations/stability_by_perturbation_thresholds_all",
    key_function=dict(mod=stable_hash, xys=stable_hash),
)
def stability_by_swing_thresholds_all(
    mod, xys, *, thresholds_yps, thresholds_swing, wiggle
):
    """
    Compute stability information for all the given x, y pairs.

    See stability_by_swing_thresholds for details.
    """
    quants, stables = [], []
    for x, y in tqdm.tqdm(xys):
        quant, stable = stability_by_swing_thresholds(
            mod,
            x,
            y,
            thresholds_yps=thresholds_yps,
            thresholds_swing=thresholds_swing,
            replicates=1000,
            seed=0,
            wiggle=wiggle,
        )
        if quant is None:
            continue
        quants.append(quant)
        stables.append(stable)
    quants = np.array(quants)
    stables = np.array(stables)
    return quants, stables


def stability_by_swing_thresholds(
    mod, x, y, *, thresholds_yps, thresholds_swing, replicates=10_000, seed, wiggle
):
    """

    Parameters:
        mod: The model to use.
        x: The x value to use.
        y: The y value to use (scalar)
        thresholds_yps: The thresholds to use to determine correctness/incorrectness.
        thresholds_swing: The thresholds to use to determine whether something should be randomized
            or not.
        replicates: The number of replicates to use.
        seed: The seed to use for the random number generator.
        wiggle: The amount of wiggle room to use (if the prediction is less than this + the threshold, we return None)

    Returns (quantiles, stabilities)
        each quantile represents the fraction of positions we will randomize.
        each stabilities is the probability that our prediction changes
    """
    orig, modified = probabilities_by_changed_sequence(
        mod, x, y, change_window=1, pbar=tqdm.tqdm
    )
    if orig < thresholds_yps[y - 1] + wiggle:
        return None, None
    swings = np.abs(modified - orig).max(-1)
    quantiles = np.array(
        [(swings < threshold).mean() for threshold in thresholds_swing]
    )
    stabilities = np.array(
        [
            (
                randomized_predictions(
                    mod, x, y, swings < threshold, replicates=replicates, seed=seed
                )
                >= thresholds_yps[y - 1]
            ).mean()
            for threshold in thresholds_swing
        ]
    )
    return quantiles, stabilities


@permacache(
    "perturbations/randomized_predictions",
    key_function=dict(mod=stable_hash, x=stable_hash, y=stable_hash),
)
def randomized_predictions(mod, x, y, randomize, *, replicates, seed):
    """
    Produce randomized predictions

    Parameters:
        mod: The model to use.
        x: (L, 4) The unmodified x value to use.
        y: (L,) The y value to select for each output x
        randomize: (L,) A boolean array of whether to randomize each position.
        replicates: The number of random replicates to use.
        seed: The seed to use for the random number generator.

    Returns:
        out (replicates, L - mod.cl) The predictions on the randomized positions.
    """
    x_wind_random = np.repeat(x[None], replicates, axis=0)
    x_wind_random[:, randomize] = np.eye(4)[
        np.random.RandomState(seed).choice(4, size=(replicates, randomize.sum()))
    ]
    return run_batched(lambda x: mod(x).softmax(-1)[:, :, y], x_wind_random, 1000)
