from sys import argv
from msp import MSP, rbnsp_split_90

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])
msp.architecture["num_motifs"] = 4
msp.architecture["motif_model_spec"] = dict(
    type="PSAMMotifModel",
    motif_spec=dict(type="rbns"),
    exclude_names=("3P", "5P") + tuple(x for x in rbnsp_split_90["exclude"] + rbnsp_split_90["include"] if x != "TRA2A"),
)
msp.architecture["sparse_spec"] = dict(type="SpatiallySparseAcrossChannels")
msp.architecture["affine_sparsity_enforcer"] = True
msp.acc_thresh = 34.3
msp.run()
