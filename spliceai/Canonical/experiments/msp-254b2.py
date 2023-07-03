from sys import argv
from msp import MSP

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])
msp.architecture = dict(
    type="PreprocessedSpliceAI",
    preprocessor_spec=dict(type="Identity"),
    spliceai_spec=dict(type="SpliceAIModule", output_size=3),
)

msp.data_spec = dict(
    type="H5Dataset",
    datapoint_extractor_spec=dict(
        type="BasicDatapointExtractor",
        rewriters=[
            dict(
                type="AdditionalChannelDataRewriter",
                out_channel=["outputs", "weights"],
                data_provider_spec=dict(
                    type="gene_length_weighting",
                    datafiles={
                        "./dataset_train_all.h5": "./datafile_train_all.h5",
                        "./dataset_test_0.h5": "./datafile_test_0.h5",
                    },
                    sl=5000,
                ),
            )
        ],
    ),
    post_processor_spec=dict(type="IdentityPostProcessor"),
)

msp.window = 10_000
msp.acc_thresh = 100

msp.run()
