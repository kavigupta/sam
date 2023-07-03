from modular_splicing.models_for_testing.models_for_testing import (
    EndToEndModelsForTesting,
)


def create_series(name, path, donor_seed, model_seeds, density):
    return EndToEndModelsForTesting(
        name=f"{name} d={donor_seed}",
        path_prefix=path,
        seeds=[x * 1000 + donor_seed for x in model_seeds],
        density=density,
    )


am_multiple_donors = [
    create_series(
        name="AM",
        path="model/msp-263b5",
        donor_seed=seed,
        model_seeds=[1, 2, 3],
        density=7.508e-2,
    )
    for seed in (2, 3)
]

lm_multiple_donors = [
    create_series(
        name="4LM",
        path="model/msp-263a5",
        donor_seed=seed,
        model_seeds=[1, 2, 3],
        density=0.178e-2,
    )
    for seed in (2,)
]
