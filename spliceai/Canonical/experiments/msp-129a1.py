from sys import argv
from msp import MSP

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])
msp.architecture["num_motifs"] = 56
msp.architecture["sparse_spec"] = dict(type="SpatiallySparseAcrossChannels")
msp.architecture["motif_model_spec"] = dict(
    type="AdjustedMotifModel",
    model_to_adjust_spec=dict(
        type="PSAMMotifModel",
        motif_spec=dict(type="rbns"),
        exclude_names=("3P", "5P") + tuple(['PUM1', 'UNK', 'RBM41', 'RBM23', 'MSI1', 'FUBP3', 'PCBP4', 'IGF2BP2', 'MBNL1', 'PCBP1', 'ESRP1', 'RBMS2', 'EWSR1', 'CELF1', 'SF1', 'HNRNPH2', 'RBM22', 'CPEB1', 'HNRNPC', 'PUF60', 'KHDRBS3', 'SNRPA', 'RBFOX3', 'RBFOX2', 'ZFP36']),
    ),
    adjustment_model_spec=msp.architecture["motif_model_spec"],
    sparsity_enforcer_spec=dict(
        type="SparsityEnforcer", sparse_spec=msp.architecture["sparse_spec"]
    ),
)
msp.architecture["affine_sparsity_enforcer"] = True

msp.run()
