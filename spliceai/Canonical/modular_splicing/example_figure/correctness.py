import numpy as np

from .run_models import attach_motif_predictions, compute_thresholds, limit

KEY_BY_NAME = dict(
    LSSI=lambda x: x["spl"], FM=lambda x: x["FM"]["res"], AM=lambda x: x["AM"]["res"]
)


def ensure_several_correctness_values(exons, models, *ensure_correctness, species):
    """
    Ensure several correctness values for the given exons.

    Also calculates the model's outputs when necessary.

    Arguments
    ---------
    exons: list of dictionaries. The exons to filter.
    models: dict[name] -> torch model. The models to use to compute the thresholds.
    ensure_correctness: list of tuples (name, ensure, mult)
        name: string. The name of the model to use.
        ensure: see `ensure_correctness_by_threshold` for more details
        mult: float. The multiplier to use for the threshold before filtering.
    species: string. The species whose data to use for computing the thresholds.
    """
    calculated_outputs = False
    for key, ensure, mult in ensure_correctness:
        if not calculated_outputs and key in {"FM", "AM"}:
            exons = limit(exons)
            exons = attach_motif_predictions(
                {"FM": models["FM"], "AM": models["AM"]}, exons
            )
            calculated_outputs = True

        exons = ensure_correctness_by_threshold(
            exons,
            KEY_BY_NAME[key],
            models[key],
            species,
            ensure=ensure,
            threshold_multiplier=mult,
        )
        print(f"{key} is {ensure}: {len(exons)}")
    return exons


def ensure_correctness_by_threshold(
    exons, key, model, species, *, ensure, threshold_multiplier=1
):
    """
    Returns the exons that meet the correctness criterion,
        according to the given model and threshold.

    Arguments
    ---------
    exons: list of dictionaries. The exons to filter.
    key: function from dictionary to array of scores. The key to use for filtering.
    model: torch model. The model to use to compute the thresholds. Should be
        the same model that was used to compute the scores.
    species: string. The species whose data to use for computing the thresholds.
    ensure: either
        - "correct": the exon must be correctly predicted by the model
        - "incorrect": at least one splicepoint must be incorrectly predicted by the model
            (False positives or false negatives are both okay)
        - "incorrect_without_subset": there must be at least one false positive and
            at least one false negative
    threshold_multiplier: float. The multiplier to use for the threshold before
        filtering. E.g., a multiplier of 2 changes the model to be twice as permissive
        in letting splicepoints through.
    """
    thresholds = compute_thresholds(model, species)
    results = []
    for ex in exons:
        yp = key(ex) >= np.log(thresholds) * threshold_multiplier
        y = np.eye(3, dtype=np.bool)[ex["y"]][:, 1:]
        if dict(
            correct=correct,
            incorrect=incorrect,
            incorrect_without_subset=incorrect_without_subset,
        )[ensure](y, yp).all():
            results.append(ex)
    return results


def correct(y, yp):
    return (yp == y).all()


def incorrect(y, yp):
    return (yp != y).any()


def incorrect_without_subset(y, yp):
    return (yp & ~y).any() and (~yp & y).any()
