from sys import argv
from msp import MSP

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])

selected_splicepoints = [
    dict(
        type="FromLoadedSplicepointModel",
        splicepoint_model_path=f"model/msp-216a1_{i}",
        splicepoint_index=2,
    )
    for i in (2, 3, 4, 5)
]


msp.architecture["num_motifs"] = len(selected_splicepoints) + 82
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
    type="ParallelMotifModels",
    specs=[*selected_splicepoints, msp.architecture["motif_model_spec"]],
    num_motifs_each=[1] * len(selected_splicepoints) + [80],
)

msp.architecture["sparse_spec"] = dict(
    type="ParallelSpatiallySparse",
    sparse_specs=[
        dict(type="SpatiallySparseAcrossChannels", sparsity=1 - 1.7e-2),
        dict(type="SpatiallySparseAcrossChannels", sparsity=0.5),
    ],
    num_channels_each=[len(selected_splicepoints), 80],
    update_indices=[1],
    get_index=1,
)

msp.architecture["affine_sparsity_enforcer"] = True

msp.acc_thresh = 80

msp.run()
