from sys import argv
from msp import MSP

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])

msp.batch_size = 1 * msp.batch_size // 2
msp.lr = float(msp.lr) * 1 / 2

msp.architecture["num_motifs"] = 82
msp.architecture["sparse_spec"] = dict(type="SpatiallySparseAcrossChannels")

msp.architecture["motif_model_spec"] = dict(
    type="LearnedMotifModel",
    motif_width=21,
    motif_fc_layers=2,
    motif_feature_extractor_spec=dict(
        type="FCMotifFeatureExtractor", extra_compute=20, num_compute_layers=2
    ),
)

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
    sparsity_multiplier=2,
)
msp.architecture["affine_sparsity_enforcer"] = True
msp.acc_thresh = 80

msp.run()
