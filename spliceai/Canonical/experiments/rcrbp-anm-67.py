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
msp.acc_thresh = 67.7

msp.extra_params += " --learned-motif-sparsity-drop-motif-frequency 0.87"
msp.extra_params += " --learned-motif-sparsity-threshold-initial 90"

msp.n_epochs = 40

msp.run()
