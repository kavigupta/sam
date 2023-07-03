from .models_for_testing import EndToEndModelsForTesting, STANDARD_DENSITY

FM_aat = EndToEndModelsForTesting(
    name="FM",
    path_prefix="model/msp-282a1",
    seeds=(1,),
    density=STANDARD_DENSITY,
)

AM_aat = EndToEndModelsForTesting(
    name="AM",
    path_prefix="model/msp-265a1",
    seeds=(1,),
    density=STANDARD_DENSITY,
)

NFM_aat = EndToEndModelsForTesting(
    name="NFM",
    path_prefix="model/msp-291a1",
    seeds=(1, 2, 3),
    density=STANDARD_DENSITY,
)
