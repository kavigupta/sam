from sys import argv
from msp import MSP

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])

msp.architecture = dict(
    type="SpliceaiAsDownstream",
    motif_model_spec=dict(
        type="PSAMMotifModel",
        motif_spec=dict(type="rbns"),
        exclude_names=("3P", "5P"),
    ),
    spliceai_spec=dict(type="SpliceAIModule", output_size=3),
    splicepoint_model_spec=msp.architecture["splicepoint_model_spec"],
    sparsity_enforcer_spec=dict(type="SparsityEnforcer"),
    sparse_spec=dict(type="SpatiallySparseAcrossChannels"),
    num_motifs=82,
    channels=200,
    affine_sparsity_enforcer=True,
)

msp.acc_thresh = 0
msp.extra_params += " --learned-motif-sparsity-threshold-initial 85"
msp.n_epochs = 40

msp.run()
