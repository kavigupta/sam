from sys import argv
from msp import MSP

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])

msp.run_reconstruction(path="model/msp-274.790a3_1", sparsity=0.178e-2, cl=10_000)

msp.run()
