from sys import argv
from msp import MSP

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])
msp.architecture["num_motifs"] = 1 + 2
msp.architecture["channels"] = 200
msp.architecture["motif_model_spec"]["motif_width"] = 13
msp.architecture["motif_model_spec"]["motif_feature_extractor_spec"] = dict(
    type="ResidualStack",
    depth=3,
)

msp.data_dir = "../data/chenxi-synthetic-dataset-20210817/synthetic_data_with_A1CF/canonical_1_False_fixed_more_top_200_synthetic_"

msp.extra_params += " --CL_max 2000 --data_chunk_to_use all"

msp.acc_thresh = 80

msp.run()
