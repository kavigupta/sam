import itertools
import unittest
import numpy as np

from permacache import stable_hash
from modular_splicing.dataset.basic_dataset import basic_dataset
from modular_splicing.dataset.h5_dataset import H5Dataset


class RegressionTestData(unittest.TestCase):
    def test_hashes(self):
        expected = {
            "basic/[dataset_train_all.h5]": "c437ee82dcf0dddd628681978bc455a6044f57b84f9e5b25ef28679c959ea77e",
            "basic_shuf/[dataset_train_all.h5]": "052f30036d9da9a917152c92ade4f397f19809dfbcbcf8abc9fb19f3b8514a77",
            "basic_shuf_unseeded/[dataset_train_all.h5]": "1d6cf33dfc9cc11fc32ae48e215e127423b3cfc923f143d312dd9187b385c6ec",
            "basic_shuf_fully_random/[dataset_train_all.h5]": "55f30013e9138de7a254d10387bf3eb73400d2e59e3943f46e58424de2caa454",
            "basic_shuf_fully_random_indices/[dataset_train_all.h5]": "b10d7614d03d0033f82ff861e1f44065d958e69b7368274fdcb0f8c7682589e9",
            "basic_shuf_clipped/[dataset_train_all.h5]": "dfff4aa702bec60fa53d71c63d5c6aa1ce43b87e770787d5af6cadfdfefc5d4a",
            "basic_shuf_clipped_sl1000/[dataset_train_all.h5]": "54ad644ac21eec2ceea55e6dc0aea6523f6f3489512d807bd8252b73c9b5074d",
            "basic_shuf_add_input/[dataset_train_all.h5]": "6a70f1319206334c35d7431fc10a3fbb8401381c31419f3a3056062630e1a188",
            "basic_shuf_add_output/[dataset_train_all.h5]": "d350da8bee65c3a064a94bd16103634dccd9e183140420ba56cbab9ad04c404e",
            "basic_shuf_add_input_output/[dataset_train_all.h5]": "6c028498b99d57c4e395c8710f6a3112a6cdbce6ed8bcf971aebe084b8fdceb4",
            "basic_shuf_fully_random_add_input_output/[dataset_train_all.h5]": "2e4f031f0f2151fbcef40c327241be26c480170aa7d9ea8e9945251216850e19",
            "basic/[dataset_test_0.h5]": "c41dc6b7c3edb254e8bed3ce52d21683c3c3aca4f5d1dad172ce808b28a0a4a8",
            "basic_shuf/[dataset_test_0.h5]": "259b1e59c696aec438975fdb718867b52f61dfe44b48365d9cb99ff083ab0c70",
            "basic_shuf_unseeded/[dataset_test_0.h5]": "56c754c5cc31c18cd4fd4b206d82af47108ebfb6e64271fd91a46f891c611956",
            "basic_shuf_fully_random/[dataset_test_0.h5]": "523d439a5298dbdf6a051aeb19cc6a0c6ad9cac3d9fd69fd66f28b3fe51360f4",
            "basic_shuf_fully_random_indices/[dataset_test_0.h5]": "8b34c54c3d7d8dfb31d5fa558c5cfec0d15dd516f20ccff3b8cb35aebf4b82af",
            "basic_shuf_clipped/[dataset_test_0.h5]": "320305990a0f5ebf301d6be02d98f96834c53cef5c193f1a84a27619871267e6",
            "basic_shuf_clipped_sl1000/[dataset_test_0.h5]": "f4bafbea6f5829ddef8bc86dc37a81924eb66600f09a59739ee448426b8ccd80",
            "basic_shuf_add_input/[dataset_test_0.h5]": "9267be5fec78e19d53cdaf2e277a661541e7b7c282130ec3dde10bf07d3fc2cc",
            "basic_shuf_add_output/[dataset_test_0.h5]": "d7fabb11a82c6775955a2e120ea6a57ce487be0eeeae21d302d428f17b8fc2e3",
            "basic_shuf_add_input_output/[dataset_test_0.h5]": "8314cd3c62ca30dd74f2e64ae973d9052e99a5f139f2645b234effa180027116",
            "basic_shuf_fully_random_add_input_output/[dataset_test_0.h5]": "d34e7c6238194f74a890ec4717caefb4f900a6463fbcea1921d0af19bd70c386",
        }
        dsets = all_data()
        print("HI")
        actual = {}
        for dset in dsets:
            print(dset)
            actual[dset] = read_data(dsets[dset])
        print(actual)
        for dset in dsets:
            self.assertEqual(actual[dset], expected.get(dset, "NOT PRESENT"), dset)

    def test_lengths(self):
        expected = {
            "basic/[dataset_train_all.h5]": 162706,
            "basic_shuf/[dataset_train_all.h5]": 162706,
            "basic_shuf_unseeded/[dataset_train_all.h5]": 162706,
            "basic_shuf_fully_random/[dataset_train_all.h5]": 162706,
            "basic_shuf_fully_random_indices/[dataset_train_all.h5]": 162706,
            "basic_shuf_clipped/[dataset_train_all.h5]": 162706,
            "basic_shuf_clipped_sl1000/[dataset_train_all.h5]": 813530,
            "basic_shuf_add_input/[dataset_train_all.h5]": 162706,
            "basic_shuf_add_output/[dataset_train_all.h5]": 162706,
            "basic_shuf_add_input_output/[dataset_train_all.h5]": 162706,
            "basic_shuf_fully_random_add_input_output/[dataset_train_all.h5]": 162706,
            "basic/[dataset_test_0.h5]": 16505,
            "basic_shuf/[dataset_test_0.h5]": 16505,
            "basic_shuf_unseeded/[dataset_test_0.h5]": 16505,
            "basic_shuf_fully_random/[dataset_test_0.h5]": 16505,
            "basic_shuf_fully_random_indices/[dataset_test_0.h5]": 16505,
            "basic_shuf_clipped/[dataset_test_0.h5]": 16505,
            "basic_shuf_clipped_sl1000/[dataset_test_0.h5]": 82525,
            "basic_shuf_add_input/[dataset_test_0.h5]": 16505,
            "basic_shuf_add_output/[dataset_test_0.h5]": 16505,
            "basic_shuf_add_input_output/[dataset_test_0.h5]": 16505,
            "basic_shuf_fully_random_add_input_output/[dataset_test_0.h5]": 16505,
        }
        dsets = all_data()
        actual = {dset: len(dsets[dset]) for dset in dsets}
        print(actual)
        for dset in dsets:
            self.assertEqual(actual[dset], expected.get(dset, "NOT PRESENT"), dset)


def get_datasets(path):
    return {
        "basic": basic_dataset(path, 10_000, 10_000, sl=None),
        "basic_shuf": basic_dataset(
            path,
            10_000,
            10_000,
            sl=None,
            iterator_spec=dict(
                type="FastIter", shuffler_spec=dict(type="SeededShuffler", seed=0x123)
            ),
        ),
        "basic_shuf_unseeded": basic_dataset(
            path,
            10_000,
            10_000,
            sl=None,
            iterator_spec=dict(
                type="FastIter", shuffler_spec=dict(type="UnseededShuffler")
            ),
        ),
        "basic_shuf_fully_random": basic_dataset(
            path,
            10_000,
            10_000,
            sl=None,
            iterator_spec=dict(
                type="FullyRandomIter",
                shuffler_spec=dict(type="SeededShuffler", seed=0x123),
            ),
        ),
        "basic_shuf_fully_random_indices": H5Dataset(
            path=path,
            cl_max=10_000,
            cl=10_000,
            sl=None,
            iterator_spec=dict(
                type="FullyRandomIter",
                shuffler_spec=dict(type="SeededShuffler", seed=0x123),
            ),
            datapoint_extractor_spec=dict(
                type="BasicDatapointExtractor",
                rewriters=[
                    dict(
                        type="AdditionalChannelDataRewriter",
                        out_channel=["inputs", "motifs"],
                        data_provider_spec=dict(type="index_tracking"),
                    ),
                ],
            ),
            post_processor_spec=dict(
                type="FlattenerPostProcessor",
                indices=[("inputs", "x"), ("outputs", "y"), ("inputs", "motifs")],
            ),
        ),
        "basic_shuf_clipped": basic_dataset(
            path,
            cl_max=10_000,
            cl=100,
            sl=None,
            iterator_spec=dict(
                type="FastIter", shuffler_spec=dict(type="SeededShuffler", seed=0x123)
            ),
        ),
        "basic_shuf_clipped_sl1000": basic_dataset(
            path,
            cl_max=10_000,
            cl=100,
            sl=1000,
            iterator_spec=dict(
                type="FastIter", shuffler_spec=dict(type="SeededShuffler", seed=0x123)
            ),
        ),
        "basic_shuf_add_input": H5Dataset(
            path=path,
            cl=10_000,
            cl_max=10_000,
            sl=None,
            iterator_spec=dict(
                type="FastIter", shuffler_spec=dict(type="SeededShuffler", seed=0x123)
            ),
            datapoint_extractor_spec=dict(
                type="BasicDatapointExtractor",
                rewriters=[
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
                    ),
                ],
            ),
            post_processor_spec=dict(
                type="FlattenerPostProcessor",
                indices=[("inputs", "x"), ("outputs", "y"), ("inputs", "motifs")],
            ),
        ),
        "basic_shuf_add_output": H5Dataset(
            path=path,
            cl=10_000,
            cl_max=10_000,
            sl=None,
            iterator_spec=dict(
                type="FastIter", shuffler_spec=dict(type="SeededShuffler", seed=0x123)
            ),
            datapoint_extractor_spec=dict(
                type="BasicDatapointExtractor",
                rewriters=[
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
                ],
            ),
            post_processor_spec=dict(
                type="FlattenerPostProcessor",
                indices=[("inputs", "x"), ("outputs", "y")],
            ),
        ),
        "basic_shuf_add_input_output": H5Dataset(
            path=path,
            cl=10_000,
            cl_max=10_000,
            sl=None,
            iterator_spec=dict(
                type="FastIter", shuffler_spec=dict(type="SeededShuffler", seed=0x123)
            ),
            datapoint_extractor_spec=dict(
                type="BasicDatapointExtractor",
                rewriters=[
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
                    ),
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
                    ),
                ],
            ),
            post_processor_spec=dict(
                type="FlattenerPostProcessor",
                indices=[("inputs", "x"), ("outputs", "y"), ("inputs", "motifs")],
            ),
        ),
        "basic_shuf_fully_random_add_input_output": H5Dataset(
            path=path,
            cl=10_000,
            cl_max=10_000,
            sl=None,
            iterator_spec=dict(
                type="FullyRandomIter",
                shuffler_spec=dict(type="SeededShuffler", seed=0x123),
            ),
            datapoint_extractor_spec=dict(
                type="BasicDatapointExtractor",
                rewriters=[
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
                    ),
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
                    ),
                ],
            ),
            post_processor_spec=dict(
                type="FlattenerPostProcessor",
                indices=[("inputs", "x"), ("outputs", "y"), ("inputs", "motifs")],
            ),
        ),
    }


def all_data():
    overall = {}
    for p in "dataset_train_all.h5", "dataset_test_0.h5":
        overall.update({f"{k}/[{p}]": v for k, v in get_datasets(path=p).items()})
    return overall


def read_data(dset):
    np.random.seed(0xABC)
    return stable_hash(list(itertools.islice(dset, 10)))
