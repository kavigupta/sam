from sys import argv
from msp import MSP

msp = MSP()
msp.file = __file__
msp.batch_size = msp.batch_size * 2 // 3
msp.seed = int(argv[1])
msp.lr = float(msp.lr) * 0.9 ** 16 * 2 / 3
msp.architecture = dict(
    type="DiscretizeMotifModel",
    original_model_path="model/msp-203aa1_3",
    original_model_step=2315595,
    reset_adaptive_sparsity_threshold_manager=True,
)
msp.run()
