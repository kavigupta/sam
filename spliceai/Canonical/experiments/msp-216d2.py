from sys import argv
from msp import MSP

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])


msp.window = 40
msp.batch_size = 150
msp.architecture = dict(
    type="SplicePointIdentifier",
    cl=40,
    asymmetric_cl=(20, 2),
    hidden_size=100,
    n_layers=4,
)

msp.data_dir = "../data/gtex_dataset/"

msp.extra_params += " --only-train acceptor"

msp.n_epochs = 40
msp.run()
