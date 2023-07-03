def setup_as_adjusted_splice_site(msp, splicepoint_spec):
    msp.architecture["splicepoint_model_spec"] = dict(
        type="BothLSSIModels",
        acceptor="model/splicepoint-model-acceptor-1",
        donor="model/splicepoint-model-donor-1",
    )

    msp.architecture["num_motifs"] = 82
    msp.architecture["sparse_spec"] = dict(type="SpatiallySparseAcrossChannels")

    am = dict(
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

    msp.architecture["motif_model_spec"] = dict(
        type="ParallelMotifModels",
        specs=[splicepoint_spec, am],
        num_motifs_each=[1, 79],
    )

    msp.architecture["affine_sparsity_enforcer"] = True

    msp.acc_thresh = 79.0
    msp.n_epochs = 40
    msp.stop_at_density = 0.17e-2


def setup_as_adjusted_donor(msp):
    splicepoint_spec = dict(
        type="AdjustedSplicepointModel",
        splicepoint_model_path="model/splicepoint-model-donor-1",
        splicepoint_index=2,
        adjustment_model_spec=msp.architecture["motif_model_spec"],
    )
    setup_as_adjusted_splice_site(msp, splicepoint_spec)


def setup_as_adjusted_acceptor(msp):
    splicepoint_spec = dict(
        type="AdjustedSplicepointModel",
        splicepoint_model_path="model/splicepoint-model-acceptor-1",
        splicepoint_index=1,
        adjustment_model_spec=msp.architecture["motif_model_spec"],
    )
    setup_as_adjusted_splice_site(msp, splicepoint_spec)


def train_in_parallel(msp):
    msp.data_spec = dict(
        type="DuplicatedSettingH5Dataset",
        post_processor_spec=dict(type="IdentityPostProcessor"),
        datapoint_extractor_spec=dict(type="MultipleSettingDatapointExtractor"),
        num_settings=2,
    )

    msp.architecture["motif_model_spec"] = dict(
        type="ParallelMotifModels",
        specs=[dict(type="NoMotifModel"), msp.architecture["motif_model_spec"]],
        num_motifs_each=[1, msp.architecture["num_motifs"] - 2],
    )

    msp.architecture["sparse_spec"] = dict(
        type="ParallelSpatiallySparse",
        sparse_specs=[
            dict(type="NoSparsity"),
            msp.architecture["sparse_spec"],
        ],
        num_channels_each=[1, msp.architecture["num_motifs"] - 2],
        update_indices=[1],
        get_index=1,
    )

    msp.architecture["num_motifs"] += 1

    # adding 2 to the indices to leave room for LSSI
    msp.architecture.update(type="WithAndWithoutAdjustedDonor", ad_index=3, ad_indicator_index=2)
