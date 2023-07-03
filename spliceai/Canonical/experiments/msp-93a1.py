from sys import argv
from msp import MSP

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])
msp.architecture["num_motifs"] = 146
msp.architecture["motif_model_spec"] = dict(
    type="PSAMMotifModel",
    motif_spec=dict(type="rcrb"),
    exclude_names=("3P", "5P"),
)
msp.architecture["sparse_spec"] = dict(type="SpatiallySparseAcrossChannels")
msp.architecture["affine_sparsity_enforcer"] = True
msp.acc_thresh = 66.7

msp.SL = 15_000
msp.window = 1200
msp.batch_size //= 3

msp.run()
