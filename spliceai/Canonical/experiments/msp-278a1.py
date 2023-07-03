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

msp.architecture["num_motifs"] = 146
msp.architecture["sparse_spec"] = dict(type="SpatiallySparseAcrossChannels")
msp.architecture["motif_model_spec"] = dict(
    type="PSAMMotifModel",
    motif_spec=dict(type="rbrc"),
    exclude_names=("3P", "5P"),
)
msp.architecture["affine_sparsity_enforcer"] = True

msp.acc_thresh = 0
msp.extra_params += " --learned-motif-sparsity-threshold-initial 85"
msp.n_epochs = 40

msp.run()
