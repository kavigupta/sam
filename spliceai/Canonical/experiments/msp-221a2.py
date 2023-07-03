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
    sparsity_multiplier=2,
)
msp.architecture["motif_model_spec"] = dict(
    type="RobustlyAdjustedMotifs",
    adjusted_model_spec=msp.architecture["motif_model_spec"],
    randomize_after_sparse=True,
)
msp.architecture["affine_sparsity_enforcer"] = True
msp.architecture["sparsity_enforcer_spec"] = dict(
    type="MultiMotifParallelSparsityEnforcer",
    count=2,
    sparsity_enforcer_spec=dict(type="SparsityEnforcer"),
)

msp.run()
