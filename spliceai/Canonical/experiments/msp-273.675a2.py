from sys import argv
from msp import MSP

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])

msp.architecture["splicepoint_model_spec"] = dict(
    type="BothLSSIModels",
    acceptor="model/splicepoint-model-acceptor-1",
    donor="model/splicepoint-model-donor-1",
)

msp.architecture["motif_model_spec"] = dict(
    type="PSAMMotifModel",
    motif_spec=dict(type="rbns"),
    exclude_names=("3P", "5P"),
)

msp.architecture["num_motifs"] = 82
msp.architecture["sparse_spec"] = dict(type="SpatiallySparseAcrossChannels")
msp.architecture["affine_sparsity_enforcer"] = True

msp.acc_thresh = 67.5
msp.n_epochs = 40

msp.run()
