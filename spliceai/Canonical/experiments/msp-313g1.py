from sys import argv
from msp import MSP

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])
msp.architecture = dict(
    type="PreprocessedSpliceAI",
    preprocessor_spec=dict(type="Identity"),
    spliceai_spec=dict(
        type="SpliceAIModule",
        output_size=3,
        spliceai_spec=dict(
            type="SpliceAI",
            module_spec=dict(
                type="ResidualUnit",
                nonlinearity_spec=dict(
                    type="BackwardsOnlyLeakyReLU", slope_on_negatives=7e-1
                ),
            ),
        ),
    ),
)

msp.n_epochs = 10
msp.acc_thresh = 100

msp.run()
