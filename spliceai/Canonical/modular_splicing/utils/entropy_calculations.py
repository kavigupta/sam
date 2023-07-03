import numpy as np


def hbern(p):
    """
    Entropy of a Bernoulli distribution with parameter p.

    Parameters
    ----------
    p : float or array_like
        Probability of success.

    Returns
    -------
    float or array_like
        Entropy of the distribution.
    """
    return -(p * np.log(p) + (1 - p) * np.log(1 - p)) / np.log(2)


def density_for_same_entropy(num_motifs_original, density_original, num_motifs_new):
    """
    Compute the density for a different number of motifs that will result in the same
    entropy as the original number of motifs.

    effectively, num_motifs_new * hbern(density_new) = num_motifs_original * hbern(density_original)

    solves for density_new

    Works as long as num_motifs_original * hbern(density_original) < num_motifs_new. Otherwise,
        this is impossible as the entropy of the new distribution will necessarily be lower
        than the original.

    Parameters:
    ----------
    num_motifs_original : int
        Number of motifs in the original model.
    density_original : float
        Sparsity of the original model.
    num_motifs_new : int
        Number of motifs in the new model.

    Returns:
    -------
    float
        Sparsity of the new model.
    """

    if num_motifs_original * hbern(density_original) > num_motifs_new:
        raise TypeError(
            "Domain error: num_motifs_original * hbern(density_original) > num_motifs_new"
        )

    hbern_new = num_motifs_original * hbern(density_original) / num_motifs_new

    low, high = 1e-10, 0.5
    while high - low > 1e-10:
        mid = (low + high) / 2
        if hbern(mid) > hbern_new:
            high = mid
        else:
            low = mid
    return mid


def starting_point_for_different_number_motifs(
    num_motifs_original, density_original, num_motifs_new, sparsity_update
):
    """
    Compute a starting point if you want to change the number of motifs in a model,
        while hitting the same entropy point as the original model's hitting the
        given density.

    Parameters
    ----------
    num_motifs_original : int
        Number of motifs in the original model
    density_original : float
        Density of the original model
    num_motifs_new : int
        Number of motifs in the new model
    sparsity_update : float
        The sparsity update in the training process (CLI argument --learned-motif-sparsity-update)

    Returns
    -------
    float
        The starting density for the new model.
    """
    density = density_for_same_entropy(
        num_motifs_original, density_original, num_motifs_new
    )
    while density < 1:
        density /= sparsity_update
    return density * sparsity_update
