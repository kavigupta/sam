from sys import argv
from msp import MSP

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])


msp.lr = 1e-3
msp.window = 40
msp.batch_size = 150
msp.architecture = dict(
    type="SplicePointIdentifier",
    cl=40,
    asymmetric_cl=(2, 6),
    hidden_size=100,
    n_layers=5,
)

msp.extra_params += " --only-train donor"

msp.n_epochs = 10

msp.run()
