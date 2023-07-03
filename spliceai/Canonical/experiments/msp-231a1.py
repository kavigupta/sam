from sys import argv
from msp import MSP

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])
msp.architecture["num_motifs"] = 32
msp.architecture["sparse_spec"] = dict(type="SpatiallySparseAcrossChannels")

msp.architecture["motif_model_spec"]["motif_width"] = 13
msp.architecture["motif_model_spec"]["motif_feature_extractor_spec"] = dict(
    type="ResidualStack",
    depth=3,
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
msp.architecture["affine_sparsity_enforcer"] = True
msp.architecture["propagate_sparsity_spec"] = dict(type="NoSparsityPropagation")

msp.acc_thresh = 50
msp.extra_params += " --learned-motif-sparsity-threshold-initial 85"
msp.n_epochs = 40

msp.run()
