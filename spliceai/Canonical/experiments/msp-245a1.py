from sys import argv
from msp import MSP

msp = MSP()

msp.file = __file__
msp.seed = int(argv[1])
msp.architecture = dict(
    type="PreprocessedSpliceAI",
    preprocessor_spec=dict(type="Identity"),
    spliceai_spec=dict(type="SpliceAIModule", output_size=4),
)

msp.data_spec = dict(
    type="H5Dataset",
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
            ),
            dict(
                type="AdditionalChannelDataRewriter",
                out_channel=["outputs", "mask"],
                data_provider_spec=dict(
                    type="branch_site_mask",
                    datafiles={
                        "True": "datafile_train_all.h5",
                        "False": "datafile_test_0.h5",
                    },
                ),
            ),
        ],
    ),
    post_processor_spec=dict(type="IdentityPostProcessor"),
)

msp.acc_thresh = 100

msp.window = 10_000

msp.run()
