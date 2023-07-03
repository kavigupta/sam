from sys import argv
from msp import MSP

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])
msp.architecture["num_motifs"] = 82

msp.architecture["motif_model_spec"] = dict(
    type="PSAMMotifModel",
    motif_spec=dict(type="rbns"),
    exclude_names=("3P", "5P"),
)

msp.architecture["motif_model_spec"] = dict(
    type="ParallelMotifModels",
    specs=[
        msp.architecture["motif_model_spec"],
        dict(type="PassthroughMotifsFromData"),
    ],
    num_motifs_each=[79, 1],
)

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
                    type="gene_level_at_rich_additional_input",
                    datafiles={
                        "./dataset_train_all.h5": "./datafile_train_all.h5",
                        "./dataset_test_0.h5": "./datafile_test_0.h5",
                    },
                    sl=5000,
                    cl_max=10_000,
                ),
            )
        ],
    ),
    post_processor_spec=dict(type="IdentityPostProcessor"),
)

msp.acc_thresh = 0
msp.extra_params += " --learned-motif-sparsity-threshold-initial 85"
msp.n_epochs = 40

msp.run()
