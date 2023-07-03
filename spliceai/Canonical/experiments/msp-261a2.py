from sys import argv
from msp import MSP, am_one_to_3_motif_widths_above_90_shorter_donor

msp = MSP()

msp.batch_size = 5
msp.lr = float(msp.lr) / 15 * msp.batch_size

msp.file = __file__
msp.seed = int(argv[1])
msp.architecture["num_motifs"] = 82
msp.architecture["sparse_spec"] = dict(type="SpatiallySparseAcrossChannels")
msp.architecture["motif_model_spec"] = dict(
    type="AdjustedMotifModel",
    model_to_adjust_spec=dict(
        type="PSAMMotifModel",
        motif_spec=dict(type="rbns"),
        exclude_names=("3P", "5P"),
    ),
    adjustment_model_spec=dict(
        type="MotifModelMultipleWidths",
        widths=am_one_to_3_motif_widths_above_90_shorter_donor,
        width_to_architecture_spec=dict(type="force_conv_size", conv_size=3),
        motif_fc_layers=5,
    ),
    sparsity_enforcer_spec=dict(
        type="SparsityEnforcer", sparse_spec=msp.architecture["sparse_spec"]
    ),
)
msp.architecture["affine_sparsity_enforcer"] = True

msp.run()
