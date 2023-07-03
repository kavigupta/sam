from modular_splicing.models_for_testing.width_models import (
    AM_21,
    AM_17,
    AM_13,
    AM_13_more_layers,
    AM_13_secondary,
    AM_13_secondary_control,
    AM_13_flanks,
    AM_13_reprocessed_21,
    AM_13_reprocessed_45,
)

normal_spec = dict(
    type="H5Dataset",
    sl=5000,
    datapoint_extractor_spec=dict(
        type="BasicDatapointExtractor",
    ),
    post_processor_spec=dict(type="IdentityPostProcessor"),
)
secondary_structure_spec = lambda is_flipped: dict(
    type="H5Dataset",
    sl=5000,
    datapoint_extractor_spec=dict(
        type="BasicDatapointExtractor",
        rewriters=[
            dict(
                type="AdditionalChannelDataRewriter",
                out_channel=["inputs", "motifs"],
                data_provider_spec=dict(
                    type="substructure_probabilities",
                    sl=40,
                    cl=30,
                    **(
                        dict(
                            preprocess_spec=dict(type="swap_ac"),
                        )
                        if is_flipped
                        else dict()
                    )
                ),
            )
        ],
    ),
    post_processor_spec=dict(type="IdentityPostProcessor"),
)

series = [
    (AM_21, normal_spec),
    (AM_17, normal_spec),
    (AM_13, normal_spec),
    (AM_13_more_layers, normal_spec),
    (AM_13_secondary, secondary_structure_spec(False)),
    (AM_13_secondary_control, secondary_structure_spec(True)),
    (AM_13_flanks, normal_spec),
    (AM_13_reprocessed_21, normal_spec),
    (AM_13_reprocessed_45, normal_spec),
]

color_map = {
    AM_21.name: "red",
    AM_17.name: "red",
    AM_13.name: "red",
    AM_13_more_layers.name: "orange",
    AM_13_secondary.name: "#cc0",
    AM_13_secondary_control.name: "#cc0",
    AM_13_flanks.name: "#0c0",
    AM_13_reprocessed_21.name: "#0cc",
    AM_13_reprocessed_45.name: "#0cc",
}
