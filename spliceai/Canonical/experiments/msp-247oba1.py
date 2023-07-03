from sys import argv
from msp import MSP

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])
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

msp.data_dir = "../data/canonical_and_gtex_dataset/"

msp.data_spec = dict(
    type="NonConflictingAlternativeDataset",
    post_processor_spec=dict(type="IdentityPostProcessor"),
    underlying_ordering=["spliceai_canonical", "spliceai_gtex"],
    outcome_to_pick="spliceai_gtex",
    channels_per_outcome=3,
    mask_channel_offsets=[1, 2],
)

msp.evaluation_criterion_spec = dict(
    type="MultiEvaluationCriterion",
    num_channels_per_prediction=3,
    num_predictions=1,
    eval_indices=[0, 1],
)

msp.acc_thresh = 0
msp.extra_params += " --learned-motif-sparsity-threshold-initial 85"
msp.n_epochs = 40

msp.run()
