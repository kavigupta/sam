import fire
import numpy as np
import tqdm

from shelved.single_perturbed.analysis import (
    spm_models,
    perturbations_for_model,
    single_perturbed_models,
)


NUM_SPLITS = 4

sl = 1000


def get_params():
    for _, original, path in spm_models():
        yield from dict(single_perturbed_models(path)).values()


def main(*, split):
    assert split in range(NUM_SPLITS)

    params = sorted(get_params())

    np.random.RandomState(0).shuffle(params)

    params = params[split::NUM_SPLITS]
    print(params)
    for p, max_step in tqdm.tqdm(params):
        print(p)
        perturbations_for_model(p=p, max_step=max_step, sl=sl)


fire.Fire(main)
