from .models_for_testing import EndToEndModelsForTesting, STANDARD_DENSITY


def width_experiment(name, path):
    return EndToEndModelsForTesting(
        name=name,
        path_prefix=f"model/{path}",
        seeds=(1, 2, 3),
        density=STANDARD_DENSITY,
    )


# all of these use aat
AM_21 = width_experiment("AM 21", "msp-265a1")
AM_17 = width_experiment("AM 17", "msp-265c1")
AM_13 = width_experiment("AM 13", "msp-265e1")
AM_13_more_layers = width_experiment("AM 13 + more layers", "msp-265ea1")
AM_13_secondary = width_experiment("AM 13 + secondary struct.", "msp-265eb1")
AM_13_secondary_control = width_experiment(
    "AM 13 + secondary struct. [control]", "msp-265ec1"
)
AM_13_flanks = width_experiment("AM 13 + flanks to 21", "msp-265ed1")
AM_13_reprocessed_21 = width_experiment(
    "AM 13 + reprocessed (w=21 total)", "msp-265ef1"
)
AM_13_reprocessed_45 = width_experiment(
    "AM 13 + reprocessed (w=37 total)", "msp-265ee1"
)
