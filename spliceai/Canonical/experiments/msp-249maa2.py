from sys import argv
from msp import MSP

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])
msp.architecture = dict(
    type="PreprocessedSpliceAI",
    preprocessor_spec=dict(type="Identity"),
    spliceai_spec=dict(type="SpliceAIModule", output_size=6),
)

msp.data_dir = "../data/canonical_and_gtex_dataset/"

msp.data_spec = dict(
    type="UnifiedAlternativeDataset",
    post_processor_spec=dict(type="IdentityPostProcessor"),
    underlying_ordering=["spliceai_canonical", "spliceai_gtex"],
)

msp.evaluation_criterion_spec = dict(
    type="MultiEvaluationCriterion", num_channels_per_prediction=3, num_predictions=2
)

msp.acc_thresh = 100
msp.n_epochs = 40

msp.run()
