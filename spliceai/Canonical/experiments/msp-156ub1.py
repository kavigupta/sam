from sys import argv
from msp import MSP

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])
msp.architecture["num_motifs"] = 2 + 2
msp.architecture["channels"] = 200
msp.architecture["motif_model_spec"]["motif_width"] = 9
msp.architecture["motif_model_spec"]["motif_feature_extractor_spec"] = dict(
    type="FCMotifFeatureExtractor",
    extra_compute=10,
    num_compute_layers=1
)

msp.data_dir = "../data/chenxi-synthetic-dataset-20210817/synthetic_data_with_A1CF/canonical_1_False_None_None_200_check_existence_synthetic_"

msp.extra_params += " --CL_max 2000 --data_chunk_to_use all"

msp.acc_thresh = 50

msp.run()
