import numpy as np

from .hsmm_model import HSMM


def fill_gaps_in_probability(p):
    [nonzero_indices] = np.nonzero(p)
    starts, ends = nonzero_indices[:-1], nonzero_indices[1:]
    gap_size = ends - starts > 1
    starts, ends = starts[gap_size], ends[gap_size]
    pnew = p.copy()
    for start, end in zip(starts, ends):
        pnew[start + 1 : end] = pnew[[start, end]].min()
    adjustment_factor = pnew.sum() / p.sum()
    pnew /= adjustment_factor
    return pnew, adjustment_factor


def fill_gaps(hsmm):
    distance_distributions = {}
    adjustment_factors = {}
    for state in hsmm.states:
        p = np.array(hsmm.distance_distributions[state], dtype=np.float64)
        pnew, adjustment_factor = fill_gaps_in_probability(p)
        distance_distributions[state] = pnew
        adjustment_factors[state] = adjustment_factor
    return (
        HSMM(
            hsmm.initial,
            distance_distributions,
            hsmm.transition_distributions,
        ),
        adjustment_factors,
    )
