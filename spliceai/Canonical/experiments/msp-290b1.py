from sys import argv
from msp import MSP

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])

msp.architecture["splicepoint_model_spec"] = dict(
    type="BothLSSIModels",
    acceptor="model/splicepoint-model-acceptor-1",
    donor="model/splicepoint-model-donor-1",
)

msp.architecture["motif_model_spec"]["motif_width"] = 21
msp.architecture["motif_model_spec"]["motif_feature_extractor_spec"] = dict(
    type="ResidualStack",
    depth=5,
)

am_channels, extra_splicepoint_channels = 80, 3

msp.architecture["num_motifs"] = am_channels + extra_splicepoint_channels + 2
msp.architecture["sparse_spec"] = dict(type="SpatiallySparseAcrossChannels")
msp.architecture["motif_model_spec"] = dict(
    type="AdjustedMotifModel",
    model_to_adjust_spec=dict(
        type="PSAMMotifModel",
        motif_spec=dict(type="rbns"),
        exclude_names=("3P", "5P"),
    ),
    adjustment_model_spec=msp.architecture["motif_model_spec"],
    sparsity_enforcer_spec=dict(
        type="SparsityEnforcer", sparse_spec=msp.architecture["sparse_spec"]
    ),
)

extra_splicepoints = dict(
    type="MultipleSplicepointModels",
    splicepoint_model_path="model/splicepoint-model-donor-1",
    splicepoint_model_channel=2,
    splicepoint_splitter_spec=dict(type="AsymmetricLinearSplicepointSplitter"),
)

msp.architecture["motif_model_spec"] = dict(
    type="ParallelMotifModels",
    specs=[msp.architecture["motif_model_spec"], extra_splicepoints],
    num_motifs_each=[am_channels, extra_splicepoint_channels],
)

msp.architecture["sparse_spec"] = dict(
    type="ParallelSpatiallySparse",
    sparse_specs=[
        dict(type="SpatiallySparseAcrossChannels"),
        dict(type="SpatiallySparseAcrossChannels", sparsity=0),
    ],
    num_channels_each=[am_channels, extra_splicepoint_channels],
    update_indices=[0],
    get_index=0,
)


msp.architecture["affine_sparsity_enforcer"] = True

msp.acc_thresh = 0
msp.extra_params += " --learned-motif-sparsity-threshold-initial 85"

msp.run()
