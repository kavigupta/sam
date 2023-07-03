from sys import argv
from msp import MSP

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])
msp.architecture["num_motifs"] = 82
msp.architecture["sparse_spec"] = dict(type="SpatiallySparseAcrossChannels")

msp.architecture["motif_model_spec"]["motif_width"] = 13
msp.architecture["motif_model_spec"]["motif_feature_extractor_spec"] = dict(
    type="ResidualStack",
    depth=3,
)

msp.architecture["motif_model_spec"] = dict(
    type="ReprocessedMotifModel",
    motif_model_1_spec=dict(
        type="PretrainedMotifModel",
        model_path="model/msp-163aa1_1",
        model_step=2428095,
        finetunable=False,
    ),
    reprocessor_spec=dict(
        type="ResidualStackSRMP",
        radius=12,
        channels=200,
        depth=3,
        influence_propagation_spec=dict(type="PropagateSparsityAndSum"),
    ),
)
msp.architecture["affine_sparsity_enforcer"] = True
msp.architecture["propagate_sparsity_spec"] = dict(type="NoSparsityPropagation")

msp.acc_thresh = 80

msp.run()
