from sys import argv
from msp import MSP

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])

msp.batch_size = 10

msp.architecture["splicepoint_model_spec"] = dict(
    type="BothLSSIModels",
    acceptor="model/splicepoint-model-acceptor-1",
    donor="model/splicepoint-model-donor-1",
)

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

msp.architecture = dict(
    type="SpliceaiAsDownstream",
    motif_model_spec=msp.architecture["motif_model_spec"],
    spliceai_spec=dict(type="SpliceAIModule", output_size=3),
    splicepoint_model_spec=msp.architecture["splicepoint_model_spec"],
    sparsity_enforcer_spec=dict(type="SparsityEnforcer"),
    sparse_spec=dict(type="SpatiallySparseAcrossChannels"),
    num_motifs=82,
    channels=200,
    affine_sparsity_enforcer=True,
)

msp.window = 10_000

msp.acc_thresh = 83.5
msp.n_epochs = 40

msp.run()
