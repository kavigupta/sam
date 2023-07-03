from .models_for_testing import TEN_EPOCHS, ModelForTestingWithStep

reconstructed_am = ModelForTestingWithStep(
    name_prefix="Reconstructed AM",
    path_prefix="model/msp-284a1",
    seed=1,
    step_value=TEN_EPOCHS[15],
)
