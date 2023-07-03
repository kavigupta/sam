OVERALL_SIZE_CANO_TEST_SET = 16505

def get_tissue_groups(name):
    import sys

    sys.path.insert(0, ".")
    from modular_splicing.gtex_data.experiments import tissue_groups

    return getattr(tissue_groups, name)


def get_path(name):
    import sys

    sys.path.insert(0, ".")
    from modular_splicing.gtex_data.experiments import create_dataset

    return create_dataset.DATASETS[name]


def setup_as_gtex(msp, tissue_group_name, dataset_name):
    tissue_groups = get_tissue_groups(tissue_group_name)

    msp.data_dir = get_path(dataset_name).data_path_folder
    msp.data_spec = dict(
        type="MultiTissueProbabilitiesH5Dataset",
        post_processor_spec=dict(type="IdentityPostProcessor"),
        datapoint_extractor_spec=dict(
            type="MultipleSettingDatapointExtractor", run_argmax=False
        ),
        tissue_groups=tissue_groups,
    )

    msp.evaluation_criterion_spec = dict(
        type="MultiEvaluationCriterion",
        num_channels_per_prediction=3,
        num_predictions=1,
        eval_indices=[0, 1],
    )
    return tissue_groups

def change_adaptive_sparsity_speed_for_gtex(msp, tissue_groups):
    t = len(tissue_groups)
    msp.decay_per_epoch = 0.9 ** t
    msp.extra_params += f" --learned-motif-sparsity-threshold-decrease-per-epoch {t}"

def mix_in_cano(msp, tissue_groups):
    pps = msp.data_spec.pop("post_processor_spec")
    msp.data_spec = dict(
        type="ConcatenateDatasets",
        specs=[
            dict(
                type="DuplicatedSettingH5Dataset",
                datapoint_extractor_spec=dict(
                    type="MultipleSettingDatapointExtractor", run_argmax=False
                ),
                num_settings=len(tissue_groups),
            ),
            msp.data_spec,
        ],
        path_replacement_specs=[
            dict(type="regex_replace", regex=msp.data_dir, replacement="./"),
            dict(type="identity"),
        ],
        post_processor_spec=pps,
    )

    # only evaluate on the first half of the cano set, for consistency.
    msp.extra_params += f" --eval-limit {int(OVERALL_SIZE_CANO_TEST_SET * 0.5)}"
