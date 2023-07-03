import tqdm.auto as tqdm
import numpy as np


def mcmc_step(state, current_score, rng, *, report, temperature=1, **kwargs):
    mutation = state.sample_mutation(rng, **kwargs)
    state.perform(mutation)
    new_score = state.score()
    log_ratio = temperature * (new_score - current_score)
    if np.log(rng.rand()) < log_ratio:
        if current_score != new_score:
            report(new_score)
        current_score = new_score
    else:
        state.undo(mutation)
    return current_score


def mcmc(state, rng, *, report, n_steps, **kwargs):
    current_score = state.score()
    for _ in tqdm.trange(n_steps):
        current_score = mcmc_step(state, current_score, rng, report=report, **kwargs)
