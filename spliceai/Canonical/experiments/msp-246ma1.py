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
msp.architecture["sparse_spec"] = dict(type="SpatiallySparseAcrossChannels")
msp.architecture["affine_sparsity_enforcer"] = True
msp.acc_thresh = 66.7

msp.data_dir = "../data/canonical_and_gtex_dataset/"

msp.data_spec = dict(
    type="UnifiedAlternativeDataset",
    post_processor_spec=dict(type="IdentityPostProcessor"),
    underlying_ordering=["spliceai_canonical", "spliceai_gtex"],
)

msp.architecture["lsmrp_spec"] = dict(type="MultiOutputLSMRP")
msp.architecture["output_size"] = 6

msp.evaluation_criterion_spec = dict(
    type="MultiEvaluationCriterion", num_channels_per_prediction=3, num_predictions=2
)

msp.run()
