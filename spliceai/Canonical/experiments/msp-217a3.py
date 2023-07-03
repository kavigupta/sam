from sys import argv
from msp import MSP

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])
msp.architecture["num_motifs"] = 82
msp.architecture["sparse_spec"] = dict(
    type="ParallelSpatiallySparse",
    sparse_specs=[
        dict(type="SpatiallySparseAcrossChannels"),
        dict(type="SpatiallySparseAcrossChannels", sparsity=1 - 1.7e-2),
    ],
    num_channels_each=[79, 1],
    update_indices=[0],
    get_index=0,
)


msp.architecture["motif_model_spec"]["motif_width"] = 13
msp.architecture["motif_model_spec"]["motif_feature_extractor_spec"] = dict(
    type="ResidualStack",
    depth=3,
)

msp.architecture["motif_model_spec"] = dict(
    type="AdjustedAlternateDonorSeparatingMotifModel",
    am_spec=dict(
        type="AdjustedMotifModel",
        adjustment_model_spec=msp.architecture["motif_model_spec"],
        sparsity_enforcer_spec=dict(
            type="SparsityEnforcer",
            sparse_spec=dict(type="SpatiallySparseAcrossChannels"),
        ),
        sparsity_multiplier=4,
    ),
    alternate_donor_separating_spec=dict(
        type="AlternateDonorSeparatingMotifModel",
        underlying_motif_model_spec=dict(
            type="PSAMMotifModel",
            motif_spec=dict(type="rbns"),
            exclude_names=("3P", "5P"),
        ),
        original_donor_spec=dict(
            type="BothLSSIModels",
            acceptor="model/splicepoint-model-acceptor-1",
            donor="model/splicepoint-donor2-2.sh",
        ),
        donor_sparsity_multiplier=1,
        suppression_offsets=list(range(-2, 10 + 1)),
        mask_above_threshold=10,
        shift_donor_amount=3,
    ),
)
msp.architecture["affine_sparsity_enforcer"] = True
msp.architecture["propagate_sparsity_spec"] = dict(type="NoSparsityPropagation")

msp.acc_thresh = 80

msp.run()
