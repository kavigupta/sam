from sys import argv
from msp import MSP

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])

msp.decay_per_epoch = 0.9 ** (5509 / 162706)

msp.lr = 1e-3
msp.window = 40
msp.batch_size = 32
msp.architecture = dict(
    type="SplicePointIdentifier",
    cl=40,
    asymmetric_cl=(20, 2),
    hidden_size=100,
    n_layers=4,
)

msp.extra_params += " --only-train acceptor"

msp.data_dir = "../data/by-length/below-10k/"
msp.n_epochs = 10000

msp.run()
