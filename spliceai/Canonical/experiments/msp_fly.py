def fly_am(msp):

    fraction_of_standard_epoch = 15294 / 162706
    msp.decay_per_epoch = 0.9 ** fraction_of_standard_epoch
    msp.architecture["num_motifs"] = 53
    msp.architecture["sparse_spec"] = dict(type="SpatiallySparseAcrossChannels")
    msp.architecture["motif_model_spec"] = dict(
        type="AdjustedMotifModel",
        model_to_adjust_spec=dict(
            type="PSAMMotifModel",
            motif_spec=dict(
                type="rna_compete_motifs_for", species_list=["Drosophila_melanogaster"]
            ),
            exclude_names=("3P", "5P"),
        ),
        adjustment_model_spec=msp.architecture["motif_model_spec"],
        sparsity_enforcer_spec=dict(
            type="SparsityEnforcer", sparse_spec=msp.architecture["sparse_spec"]
        ),
    )
    msp.architecture["splicepoint_model_spec"] = dict(
        type="BothLSSIModels",
        acceptor="../rbns_model/splicepoint_model/model_acceptor/0",
        donor="../rbns_model/splicepoint_model/model_donor/0",
    )
    msp.architecture["affine_sparsity_enforcer"] = True

    msp.data_dir = "../rbns_model/organism/drosophila/10k_updated/drosophila_"

    msp.acc_thresh = 0
    msp.extra_params += " --learned-motif-sparsity-threshold-initial 90"
    msp.n_epochs = 1000

    msp.extra_params += f" --learned-motif-sparsity-threshold-decrease-per-epoch {fraction_of_standard_epoch}"

    msp.stop_at_density = 0.3e-2
