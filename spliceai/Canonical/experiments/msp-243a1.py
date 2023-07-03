from sys import argv
from msp import MSP

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])
msp.architecture["num_motifs"] = 80
msp.architecture["sparse_spec"] = dict(type="SpatiallySparseAcrossChannels")

msp.architecture["motif_model_spec"] = dict(type="NoMotifModel")
msp.architecture["affine_sparsity_enforcer"] = True

msp.acc_thresh = 100
msp.n_epochs = 40

msp.run()
