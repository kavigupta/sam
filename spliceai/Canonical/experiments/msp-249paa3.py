from sys import argv
from msp import MSP

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])
msp.architecture = dict(
    type="PreprocessedSpliceAI",
    preprocessor_spec=dict(type="Identity"),
    spliceai_spec=dict(type="SpliceAIModule", output_size=3),
)

msp.data_dir = "../data/canonical_and_gtex_dataset/"

msp.data_spec = dict(
    type="NonConflictingAlternativeDataset",
    post_processor_spec=dict(type="IdentityPostProcessor"),
    underlying_ordering=["spliceai_canonical", "spliceai_gtex"],
    outcome_to_pick="spliceai_canonical",
    channels_per_outcome=3,
    mask_channel_offsets=[1, 2],
    always_keep_picked=True,
)

msp.evaluation_criterion_spec = dict(
    type="MultiEvaluationCriterion",
    num_channels_per_prediction=3,
    num_predictions=1,
    eval_indices=[0, 1],
)

msp.acc_thresh = 100
msp.n_epochs = 40

msp.run()
