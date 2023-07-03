from sys import argv
from msp import MSP

msp = MSP()
msp.file = __file__
msp.batch_size = msp.batch_size * 2 // 3
msp.seed = int(argv[1])
msp.lr = float(msp.lr) * 0.9 ** 16 * 2 / 3
msp.architecture = dict(
    type="DroppedMotifFromModel",
    original_model_path="model/msp-8a2_1",
    original_model_step=4010295,
    dropped_motifs=[20],
)
msp.run()
