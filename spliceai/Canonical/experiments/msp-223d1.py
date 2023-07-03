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
    type="AdjustedMotifModel",
    model_to_adjust_spec=dict(
        type="PSAMMotifModel",
        motif_spec=dict(type="rbrc"),
        include_names=[
            "CPEB4",
            "EIF4G2",
            "EWSR1",
            "FMR1",
            "FUS",
            "FXR1",
            "FXR2",
            "HNRNPA1",
            "HNRNPC",
            "HNRNPK",
            "HNRNPL",
            "IGF2BP1",
            "IGF2BP2",
            "KHDRBS1",
            "KHSRP",
            "MATR3",
            "PABPC4",
            "PCBP1",
            "PTBP1",
            "PUM1",
            "QKI",
            "RBFOX2",
            "RBM22",
            "SRSF1",
            "SRSF7",
            "TAF15",
            "TARDBP",
            "TIA1",
            "TRA2A",
            "U2AF2",
        ],
    ),
    adjustment_model_spec=msp.architecture["motif_model_spec"],
    sparsity_enforcer_spec=dict(
        type="SparsityEnforcer", sparse_spec=msp.architecture["sparse_spec"]
    ),
    sparsity_multiplier=2,
)
msp.architecture["affine_sparsity_enforcer"] = True
msp.architecture["propagate_sparsity_spec"] = dict(type="NoSparsityPropagation")

msp.acc_thresh = 50
msp.extra_params += " --learned-motif-sparsity-threshold-initial 85"
msp.n_epochs = 40

msp.run()
