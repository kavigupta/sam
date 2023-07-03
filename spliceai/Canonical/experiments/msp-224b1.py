from sys import argv
from msp import MSP

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])
msp.sparse_adaptation_params = lambda: ""
msp.architecture = dict(
    type="PreprocessedSpliceAI",
    preprocessor_spec=dict(
        type="Motifs",
        motifs_spec=dict(
            type="PSAMMotifModel",
            motif_spec=dict(type="rbrc"),
            include_names=["TRA2A"],
            channels=100,
            num_motifs=1,
        ),
    ),
    spliceai_spec=dict(type="SpliceAIModule"),
    splicepoint_model_spec=dict(
        type="BothLSSIModels",
        acceptor="model/splicepoint-model-acceptor-1",
        donor="model/splicepoint-donor2-2.sh",
    ),
    post_processor=1,
)

msp.data_dir = "../data/chenxi-synthetic-dataset-20211210-tra2a/synthetic_dataset_TRA2A/canonical_1_False_fixed_more_top_200_TRA2A_synthetic_"

msp.extra_params += " --CL_max 2000 --data_chunk_to_use all"

msp.run()
