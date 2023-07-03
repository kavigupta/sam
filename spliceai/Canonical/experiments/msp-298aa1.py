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

msp.architecture["num_motifs"] = 82
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
msp.architecture["affine_sparsity_enforcer"] = True

msp.architecture["final_processor_spec"].update(
    dict(
        bottleneck_spec=dict(type="ConvolutionalBottleneck", width=17),
        post_bottleneck_lsmrp_spec=dict(type="EarlyChannelLSMRP"),
        evaluate_post_bottleneck=True,
    )
)
msp.architecture["lsmrp_spec"] = dict(type="DoNotPropagateLSMRP")

msp.architecture["influence_calculator_spec"] = dict(
    type="FullTableLinearEffects",
    num_iterations=1,
)
msp.architecture["propagate_sparsity_spec"] = dict(type="JustUseInfluence")

msp.acc_thresh = 0
msp.extra_params += " --learned-motif-sparsity-threshold-initial 85"

msp.n_epochs = 40

msp.run()
