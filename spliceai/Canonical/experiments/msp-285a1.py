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

msp.architecture["motif_model_spec"] = dict(
    type="PSAMMotifModel",
    motif_spec=dict(type="rbns"),
    include_names=[
        "EIF4G2",
        "EWSR1",
        "FUS",
        "HNRNPA1",
        "HNRNPC",
        "HNRNPK",
        "HNRNPL",
        "IGF2BP1",
        "IGF2BP2",
        "KHSRP",
        "PCBP1",
        "PUM1",
        "RBFOX2",
        "RBM22",
        "TAF15",
        "TARDBP",
        "TIA1",
        "TRA2A",
    ],
)

msp.architecture["num_motifs"] = 82
msp.architecture["sparse_spec"] = dict(type="SpatiallySparseAcrossChannels")
msp.architecture["affine_sparsity_enforcer"] = True

msp.acc_thresh = 0
msp.extra_params += " --learned-motif-sparsity-threshold-initial 85"
msp.n_epochs = 40

msp.run()
