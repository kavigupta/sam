from sys import argv
from msp import MSP, setup_as_am
from msp_gtex import setup_as_gtex, change_adaptive_sparsity_speed_for_gtex

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])

setup_as_am(msp)
tissue_groups = setup_as_gtex(msp, "main_tissue_groups_v1", "MarginalPsiV3")

msp.architecture.update(
    type="MultiTissueProbabilitiesPredictorEnvPresparse",
    num_tissues=len(tissue_groups),
)

msp.acc_thresh = 0
msp.extra_params += " --learned-motif-sparsity-threshold-initial 85"

msp.stop_at_density = 0.17e-2

msp.run()
