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

msp.data_dir = "../data/gtex_derived_alt_const_split/1000/uniform/"

msp.data_spec = dict(
    type="NonConflictingAlternativeDataset",
    post_processor_spec=dict(type="IdentityPostProcessor"),
    underlying_ordering=["gtex_const_and_alt", "gtex_const"],
    outcome_to_pick="gtex_const_and_alt",
    channels_per_outcome=3,
    mask_channel_offsets=[1, 2],
    actually_mask_others=False,
)

msp.evaluation_criterion_spec = dict(
    type="MultiEvaluationCriterion",
    num_channels_per_prediction=3,
    num_predictions=1,
    eval_indices=[0, 1],
)

msp.acc_thresh = 0
msp.extra_params += " --learned-motif-sparsity-threshold-initial 85"
msp.n_epochs = 60

msp.run()
