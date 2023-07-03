import tqdm.auto as tqdm

from modular_splicing.dataset.h5_dataset import H5Dataset


def go(path):
    dset = H5Dataset(
        path=path,
        cl=400,
        cl_max=10_000,
        sl=5000,
        iterator_spec=dict(type="FastIter", shuffler_spec=dict(type="DoNotShuffle")),
        datapoint_extractor_spec=dict(
            type="BasicDatapointExtractor",
            rewriters=[
                dict(
                    type="AdditionalChannelDataRewriter",
                    out_channel=["inputs", "motifs"],
                    data_provider_spec=dict(
                        type="substructure_probabilities",
                        sl=40,
                        cl=30,
                        preprocess_spec=dict(type="swap_ac"),
                    ),
                )
            ],
        ),
        post_processor_spec=dict(type="IdentityPostProcessor"),
    )
    for _ in tqdm.tqdm(dset):
        continue


go("dataset_train_all.h5")
go("dataset_test_0.h5")
