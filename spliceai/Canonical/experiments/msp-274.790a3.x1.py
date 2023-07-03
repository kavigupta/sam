from sys import argv
from msp import MSP

msp = MSP()
msp.file = __file__
msp.batch_size = msp.batch_size * 2 // 3
msp.seed = int(argv[1])
msp.lr = float(msp.lr) * 0.9 ** 16 * 2 / 3

msp.acc_thresh = 100
msp.n_epochs = 10

msp.run_binarizer(0.178e-2)
