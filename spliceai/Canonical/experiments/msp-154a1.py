from sys import argv
from msp import MSP

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])
msp.architecture["num_motifs"] = 4 + 2
msp.architecture["motif_model_spec"] = dict(
    type="PSAMMotifModel",
    motif_spec=dict(
        type="grouped",
        original_spec=dict(type="rbns"),
        group_motifs_path="output-csvs/rbnsp-grouped-motifs-v02_4lm.json",
        remove=[
            "3P",
            "5P",
            "CNOT4",
            "EWSR1",
            "FUS",
            "HNRNPK",
            "MBNL1",
            "NOVA1",
            "PCBP1",
            "PCBP2",
            "PCBP4",
            "RBFOX3",
            "RBM23",
            "RBM6",
            "SNRPA",
        ],
    ),
    exclude_names=("3P", "5P"),
)
msp.architecture["sparse_spec"] = dict(type="SpatiallySparseAcrossChannels")
msp.architecture["affine_sparsity_enforcer"] = True
msp.acc_thresh = 80
msp.run()
