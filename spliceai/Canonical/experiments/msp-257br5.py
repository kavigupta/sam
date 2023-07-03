from sys import argv
from msp import MSP

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])

fraction_of_standard_epoch = 9241 / 162706
msp.decay_per_epoch = 0.9 ** fraction_of_standard_epoch

msp.architecture["num_motifs"] = 82
msp.architecture["sparse_spec"] = dict(type="SpatiallySparseAcrossChannels")
msp.architecture["motif_model_spec"] = dict(
    type="AdjustedMotifModel",
    model_to_adjust_spec=dict(
        type="PSAMMotifModel",
        motif_spec=dict(type="rbns"),
        exclude_names=("3P", "5P"),
    ),
    adjustment_model_spec=msp.architecture["motif_model_spec"],
    sparsity_enforcer_spec=dict(
        type="SparsityEnforcer", sparse_spec=msp.architecture["sparse_spec"]
    ),
)
msp.architecture["affine_sparsity_enforcer"] = True


msp.data_dir = "../data/by-length/balanced-at-10k/"

msp.acc_thresh = 0
msp.extra_params += (
    " --learned-motif-sparsity-threshold-initial 85"
    + f" --learned-motif-sparsity-threshold-decrease-per-epoch {fraction_of_standard_epoch}"
)
msp.n_epochs = 4000

msp.run()
