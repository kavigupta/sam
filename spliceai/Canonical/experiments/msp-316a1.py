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

density = 0.75**22

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
        type="SparsityEnforcer",
        sparse_spec=dict(
            type="SpatiallySparseAcrossChannels", sparsity=1 - 2 * density
        ),
    ),
)

msp.architecture["affine_sparsity_enforcer"] = True

msp.architecture["sparsity_enforcer_spec"] = dict(
    type="SparsityEnforcer",
    sparse_spec=dict(
        type="SpatiallySparseAcrossChannels",
        sparsity=1 - density,
    ),
)

msp.acc_thresh = 100
msp.n_epochs = 20

msp.stop_at_density = 0.17e-2

msp.run()
