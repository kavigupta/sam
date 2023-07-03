from sys import argv
from msp import MSP

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])

fraction_of_standard_epoch = 5509 / 162706
msp.decay_per_epoch = 0.9 ** fraction_of_standard_epoch

msp.architecture["num_motifs"] = 82
msp.architecture["motif_model_spec"] = dict(
    type="PSAMMotifModel",
    motif_spec=dict(type="rbns"),
    exclude_names=("3P", "5P"),
)
msp.architecture["sparse_spec"] = dict(type="SpatiallySparseAcrossChannels")
msp.architecture["affine_sparsity_enforcer"] = True


msp.data_dir = "../data/by-length/below-10k/"

msp.acc_thresh = 0
msp.extra_params += (
    " --learned-motif-sparsity-threshold-initial 85"
    + f" --learned-motif-sparsity-threshold-decrease-per-epoch {fraction_of_standard_epoch}"
)
msp.n_epochs = 4000

msp.run()
