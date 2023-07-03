from .models_for_testing import EndToEndModelsForTesting, STANDARD_DENSITY

MD_2 = EndToEndModelsForTesting(
    name="MD_2",
    path_prefix="model/msp-290a1",
    seeds=(1, 2, 3),
    density=STANDARD_DENSITY,
)

MD_3 = EndToEndModelsForTesting(
    name="MD_3",
    path_prefix="model/msp-290b1",
    seeds=(1, 2, 3),
    density=STANDARD_DENSITY,
)
