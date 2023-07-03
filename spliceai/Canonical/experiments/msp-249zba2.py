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

msp.acc_thresh = 100

msp.window = 10_000

msp.run()
