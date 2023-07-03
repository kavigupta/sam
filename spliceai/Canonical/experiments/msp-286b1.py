from sys import argv
from msp import MSP, starting_point_for_different_number_motifs

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])

msp.architecture["splicepoint_model_spec"] = dict(
    type="BothLSSIModels",
    acceptor="model/splicepoint-model-acceptor-1",
    donor="model/splicepoint-model-donor-1",
)

msp.architecture["num_motifs"] = 120
msp.architecture["sparse_spec"] = dict(type="SpatiallySparseAcrossChannels")
msp.architecture["motif_model_spec"] = dict(
    type="AdjustedMotifModel",
    model_to_adjust_spec=dict(
        type="PSAMMotifModel",
        motif_spec=dict(type="rbns_v2p1"),
        exclude_names=("3P", "5P"),
    ),
    adjustment_model_spec=msp.architecture["motif_model_spec"],
    sparsity_enforcer_spec=dict(
        type="SparsityEnforcer", sparse_spec=msp.architecture["sparse_spec"]
    ),
)
msp.architecture["affine_sparsity_enforcer"] = True

msp.acc_thresh = 0
msp.extra_params += " --learned-motif-sparsity-threshold-initial 85"
msp.n_epochs = 40
msp.sparsity_start = starting_point_for_different_number_motifs(
    79, 0.178e-2, msp.architecture["num_motifs"] - 2, 0.75
)

msp.run()
