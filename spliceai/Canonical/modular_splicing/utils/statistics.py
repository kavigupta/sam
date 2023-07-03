import numpy as np


def bootstrap(vals, n=10000, percentile=5):
    """
    Bootstrap a set of values to get a confidence interval.

    Parameters
    ----------
    vals : array_like
        Values to bootstrap.
    n : int
        Number of bootstrap samples to take.
    percentile : int
        Percentile to use for the confidence interval. E.g., a 95% confidence interval
        would be percentile=5.

    Returns
    -------
    float, float
        Lower and upper bounds of the confidence interval.
    """
    vals = np.array(vals).flatten()
    results = (
        np.random.RandomState(0).choice(vals, size=(vals.size, n), replace=True).mean(0)
    )
    return np.percentile(results, percentile / 2), np.percentile(
        results, 100 - percentile / 2
    )
