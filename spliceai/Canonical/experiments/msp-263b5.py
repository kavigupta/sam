from sys import argv
from msp import MSP

msp = MSP()

msp.batch_size = 2 * msp.batch_size // 3
msp.lr = float(msp.lr) * 2 / 3

msp.file = __file__
msp.seed = int(argv[1])
donor_seed = msp.seed % 1000

msp.architecture["splicepoint_model_spec"]["donor"] = f"model/msp-262da5_{donor_seed}"

msp.architecture["motif_model_spec"]["motif_width"] = 21
msp.architecture["motif_model_spec"]["motif_feature_extractor_spec"] = dict(
    type="ResidualStack",
    depth=5,
)

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

msp.n_epochs = 20

msp.acc_thresh = 0
msp.extra_params += " --learned-motif-sparsity-threshold-initial 85"

msp.run()
