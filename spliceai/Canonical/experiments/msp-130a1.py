from sys import argv
from msp import MSP

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])
msp.architecture["num_motifs"] = 66
msp.architecture["motif_model_spec"] = dict(
    type="PSAMMotifModel",
    motif_spec=dict(type="rbns"),
    exclude_names=("3P", "5P") + tuple(['UNK', 'RBM23', 'PUM1', 'FUBP3', 'PCBP4', 'MBNL1', 'RBMS2', 'EWSR1', 'CPEB1', 'HNRNPC', 'PUF60', 'KHDRBS3', 'SNRPA', 'RBFOX2', 'RBM22', 'PCBP1']),
)
msp.architecture["sparse_spec"] = dict(type="SpatiallySparseAcrossChannels")
msp.architecture["affine_sparsity_enforcer"] = True
msp.acc_thresh = 67.7
msp.run()
