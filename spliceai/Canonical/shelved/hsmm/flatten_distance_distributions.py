from .hsmm_model import HSMM


def flatten_hsmm(hsmm):
    assert all(v[-1] == 1 for v in hsmm.distance_distributions.values())

    to_split = {
        k: len(hsmm.distance_distributions[k])
        for k in hsmm.states
        if len(hsmm.distance_distributions[k]) > 1
    }

    new_initial = modify_flattened_inflow(to_split, hsmm.initial)

    new_transition = {}
    for s_in in hsmm.states:
        transition_out = modify_flattened_inflow(
            to_split, hsmm.transition_distributions[s_in]
        )
        if s_in in to_split:
            for idx in range(to_split[s_in] - 1):
                new_transition[("flattened", idx, s_in)] = {
                    ("flattened", idx + 1, s_in): 1
                }
            new_transition[("flattened", to_split[s_in] - 1, s_in)] = transition_out
        else:
            new_transition[s_in] = transition_out

    new_distance_distributions = {k: [1] for k in new_transition}

    return to_split, HSMM(new_initial, new_distance_distributions, new_transition)


def modify_flattened_inflow(to_split, original):
    new_initial = {}
    for k, v in original.items():
        if k in to_split:
            new_initial[("flattened", 0, k)] = v
        else:
            new_initial[k] = v
    return new_initial
