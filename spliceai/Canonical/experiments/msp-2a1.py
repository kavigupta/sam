from sys import argv
from msp import MSP

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])
msp.extra_params += " --learned-motif-sparsity-drop-motif-frequency 0.87"
msp.extra_params += " --learned-motif-sparsity-threshold-initial 90"
msp.run()
