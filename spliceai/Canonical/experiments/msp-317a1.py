from sys import argv
from msp import MSP, setup_as_am
from msp_gtex import setup_as_gtex

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])

setup_as_am(msp)
tissue_groups = setup_as_gtex(msp, "just_whole_blood", "MarginalPsiV3")
assert len(tissue_groups) == 1

msp.acc_thresh = 0
msp.extra_params += " --learned-motif-sparsity-threshold-initial 85"

msp.stop_at_density = 0.17e-2
msp.n_epochs = 1000

msp.run()
