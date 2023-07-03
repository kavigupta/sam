from sys import argv
from msp import MSP

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])
msp.architecture["num_motifs"] = 82

msp.architecture["motif_model_spec"] = dict(type="NoMotifModel")
msp.architecture["propagate_sparsity_spec"] = dict(type="NoSparsityPropagation")


msp.architecture["sparse_spec"] = dict(
    type="ParallelSpatiallySparse",
    sparse_specs=[
        dict(type="SpatiallySparseAcrossChannels"),
        dict(type="SpatiallySparseAcrossChannels", sparsity=0),
    ],
    num_channels_each=[79, 1],
    update_indices=[0],
    get_index=0,
)

msp.architecture["affine_sparsity_enforcer"] = True

msp.data_spec = dict(
    type="H5Dataset",
    datapoint_extractor_spec=dict(
        type="BasicDatapointExtractor",
        rewriters=[
            dict(
                type="AdditionalChannelDataRewriter",
                out_channel=["inputs", "motifs"],
                data_provider_spec=dict(
                    type="at_rich_additional_input",
                    chunk_width=100,
                ),
            )
        ],
    ),
    post_processor_spec=dict(type="IdentityPostProcessor"),
)

msp.run()
