from sys import argv
from msp import MSP, setup_as_am

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])

setup_as_am(msp)

msp.acc_thresh = 0
msp.extra_params += " --learned-motif-sparsity-threshold-initial 85"

msp.run()
