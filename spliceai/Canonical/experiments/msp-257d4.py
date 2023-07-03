from sys import argv
from msp import MSP

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])

fraction_of_standard_epoch = 5509 / 162706
msp.decay_per_epoch = 0.9 ** fraction_of_standard_epoch

msp.architecture = dict(
    type="PreprocessedSpliceAI",
    preprocessor_spec=dict(type="Identity"),
    spliceai_spec=dict(type="SpliceAIModule", output_size=3),
)

msp.data_dir = "../data/by-length/below-10k/"

msp.acc_thresh = 100
msp.n_epochs = 400

msp.run()
