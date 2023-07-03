from sys import argv
from msp import MSP

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])
msp.architecture["num_motifs"] = 80
msp.architecture["sparse_spec"] = dict(type="SpatiallySparseAcrossChannels")

msp.architecture["motif_model_spec"] = dict(type="NoMotifModel")
msp.architecture["affine_sparsity_enforcer"] = True

msp.architecture["motif_model_spec"] = dict(
    type="ParallelMotifModels",
    specs=[
        msp.architecture["motif_model_spec"],
        dict(type="PassthroughMotifsFromData"),
    ],
    num_motifs_each=[77, 1],
)

msp.architecture["sparse_spec"] = dict(
    type="ParallelSpatiallySparse",
    sparse_specs=[
        dict(type="SpatiallySparseAcrossChannels"),
        dict(type="SpatiallySparseAcrossChannels", sparsity=0),
    ],
    num_channels_each=[77, 1],
    update_indices=[0],
    get_index=0,
)

msp.data_spec = dict(
    type="H5Dataset",
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
                ),
            )
        ],
    ),
    post_processor_spec=dict(type="IdentityPostProcessor"),
)

msp.acc_thresh = 100
msp.n_epochs = 40

msp.run()
