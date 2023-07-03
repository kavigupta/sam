from modular_splicing.models_for_testing.load_model_for_testing import step_for_density
from modular_splicing.utils.io import load_model
from shelved.metric.fuzzy_topk import all_fuzzy_topks


def evaluate_models(
    models, *, delta_max, load_fn, require_single_branch_site, cl_for_model
):
    results = {}
    for mname in models:
        model = load_fn(models[mname])
        results[mname] = all_fuzzy_topks(
            model,
            evaluation_data_spec(
                provide_input="input" in mname,
                require_single_branch_site=require_single_branch_site,
                cl=cl_for_model(mname),
                modify_output=True,
            ),
            delta_max=delta_max,
        )
    return results


def evaluation_data_spec(
    *, provide_input, require_single_branch_site, modify_output=False, cl=400
):
    mask_data_provider = dict(
        type="branch_site_mask",
        datafiles={
            "True": "datafile_train_all.h5",
            "False": "datafile_test_0.h5",
        },
    )
    if require_single_branch_site:
        mask_data_provider["num_branch_site_filter"] = dict(type="exactly", value=1)
    rewriters = []
    if modify_output:
        rewriters += [
            dict(
                type="AdditionalChannelDataRewriter",
                out_channel=["outputs", "y"],
                data_provider_spec=dict(
                    type="branch_site",
                    datafiles={
                        "True": "datafile_train_all.h5",
                        "False": "datafile_test_0.h5",
                    },
                ),
                combinator_spec=dict(type="OneHotConcatenatingCombinator"),
            )
        ]
    rewriters += [
        dict(
            type="AdditionalChannelDataRewriter",
            out_channel=["outputs", "mask"],
            data_provider_spec=mask_data_provider,
        ),
    ]
    if provide_input:
        rewriters.append(
            dict(
                type="AdditionalChannelDataRewriter",
                out_channel=["inputs", "motifs"],
                data_provider_spec=dict(
                    type="branch_site",
                    datafiles={
                        "True": "datafile_train_all.h5",
                        "False": "datafile_test_0.h5",
                    },
                ),
            )
        )
    return dict(
        type="H5Dataset",
        path="dataset_test_0.h5",
        sl=5000,
        cl=cl,
        cl_max=10_000,
        iterator_spec=dict(type="FastIter", shuffler_spec=dict(type="DoNotShuffle")),
        datapoint_extractor_spec=dict(
            type="BasicDatapointExtractor",
            rewriters=rewriters,
        ),
        post_processor_spec=dict(type="IdentityPostProcessor"),
    )


def load_latest(path):
    return load_model(f"model/{path}")[1].eval()


def load_at_sparsity(sparsity, path):
    if path.startswith("msp-238"):
        return load_latest(path)
    path = f"model/{path}"
    return load_model(path, step_for_density(path, sparsity))[1].eval()
