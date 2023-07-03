import fire
from modular_splicing.models_for_testing.main_models import AM, FM_full
from modular_splicing.motif_perturbations.perturbations_on_standardized_sample import (
    all_mpi_on_standardized_sample,
)
from working.donorlikes import am_multiple_donors, lm_multiple_donors

main_models = FM_full.binarized_models() + AM.binarized_models()
models = dict(
    main=main_models,
    donorlikes=[
        model
        for serieses in [am_multiple_donors, lm_multiple_donors]
        for series in serieses
        for model in series.non_binarized_models()
    ],
    **{m.name: [m] for m in main_models},
)


def main(k, is_binary):
    for always_use_fm in [False, True]:
        for amt in [200, 400, 800, 1600, 3200, 6400, 12800, 32_000, 64_000]:
            print(f"always_use_fm={always_use_fm}, amt={amt}")
            all_mpi_on_standardized_sample(
                models[k],
                always_use_fm=always_use_fm,
                sl=1000,
                amount=amt,
                is_binary=is_binary,
            )


fire.Fire(main)
