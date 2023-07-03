from .models_for_testing import (
    TWENTY_EPOCHS,
    EndToEndModelsForTesting,
    ModelForTestingWithStep,
)


FM_eclip_18 = EndToEndModelsForTesting(
    name="FM_eclip_18",
    path_prefix="model/msp-285a1",
    seeds=(1,),
    # approximately 0.18e-2 * 18 / 79
    density=0.0423e-2,
)

AM_E = ModelForTestingWithStep(
    name_prefix="AM_E",
    path_prefix="model/msp-287a2",
    seed=1,
    step_value=TWENTY_EPOCHS[15],
)
