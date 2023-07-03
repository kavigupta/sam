from sys import argv
from msp import MSP
from msp_donorlike import setup_as_adjusted_acceptor, train_in_parallel

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])

setup_as_adjusted_acceptor(msp)

train_in_parallel(msp)


msp.run()
