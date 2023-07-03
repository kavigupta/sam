import numpy as np

from .hsmm_model import HSMM


def substring_hsmm(hsmm, new_initial, ell):
    return HSMM(
        initial=new_initial,
        distance_distributions=substring_p(hsmm, ell),
        transition_distributions=substring_a(hsmm, ell),
    )


def substring_a(hsmm, ell):
    states = hsmm.states
    all_gammas = {}
    for s in states:
        pj = np.array(hsmm.distance_distributions[s])
        j = np.arange(pj.size) + 1
        gamma = (np.minimum(j, ell) * pj / ell).sum()
        all_gammas[s] = gamma

    new_a = {}
    for s in states:
        new_a[("A", s)] = {}
        new_a[("B", s)] = {}
        new_a[("C", s)] = {"omega": 1}
        new_a[("F", s)] = {("F", s): 1}
        new_a["omega"] = {"omega": 1}
        for sprime in states:
            for t in "A", "B":
                new_a[(t, s)]["B", sprime] = (
                    1 - all_gammas[sprime]
                ) * hsmm.transition_distributions[s].get(sprime, 0)
                new_a[(t, s)]["C", sprime] = (
                    all_gammas[sprime]
                ) * hsmm.transition_distributions[s].get(sprime, 0)
    return {
        k1: {k2: new_a[k1][k2] for k2 in new_a[k1] if new_a[k1][k2] != 0}
        for k1 in new_a
    }


def substring_p(hsmm, ell):
    new_p = {}
    for s in hsmm.states:
        new_p["omega"] = [1]
        new_p[("F", s)] = [1]
        new_p[("B", s)] = hsmm.distance_distributions[s]
        new_p[("C", s)] = new_p[("A", s)] = cut_off_length_distribution(
            hsmm.distance_distributions[s]
        )
    for s in new_p:
        new_p[s] = np.array(new_p[s][:ell])
        new_p[s] = new_p[s] / new_p[s].sum()
    return new_p


def cut_off_length_distribution(p):
    p = np.array(p)
    p = p / (np.arange(p.size) + 1)
    return np.cumsum(p[::-1])[::-1]
