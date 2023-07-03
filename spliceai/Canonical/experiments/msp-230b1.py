from sys import argv
from msp import MSP

msp = MSP()
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
    adjustment_model_spec=msp.architecture["motif_model_spec"],
    sparsity_enforcer_spec=dict(
        type="SparsityEnforcer", sparse_spec=msp.architecture["sparse_spec"]
    ),
    sparsity_multiplier=4,
)
msp.architecture["affine_sparsity_enforcer"] = True

msp.architecture["sparsity"] = 0.75
msp.architecture["cl"] = msp.window

msp.architecture = dict(
    type="DroppedMotifFromModel",
    original_model_path=msp.architecture,
    original_model_step=None,
    dropped_motifs=[47],
)
msp.acc_thresh = 80

msp.run()
