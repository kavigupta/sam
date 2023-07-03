from sys import argv
from msp import MSP

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])
msp.architecture["num_motifs"] = 82
msp.architecture["sparse_spec"] = dict(type="SpatiallySparseAcrossChannels")

msp.architecture["motif_model_spec"] = dict(
    type="MotifModelForRobustDownstream",
    models_for_motifs=[
        ("model/msp-174a1_1", 1301680),
        ("model/msp-174b1_1", 1301680),
    ],
    trainable=False,
)

msp.architecture["sparse_spec"] = dict(type="NoSparsity")
msp.architecture["sparsity_enforcer_spec"] = dict(
    type="SparsityEnforcer",
    presparse_normalizer_spec=dict(type="NoPresparseNormalizer"),
)

msp.architecture["affine_sparsity_enforcer"] = True
msp.architecture["propagate_sparsity_spec"] = dict(type="NoSparsityPropagation")

msp.acc_thresh = 80

msp.run()
