from sys import argv
from msp import MSP

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])
msp.architecture["motif_model_spec"]["motif_width"] = 9
msp.architecture["motif_model_spec"]["motif_feature_extractor_spec"] = dict(
    type="FCMotifFeatureExtractor",
    extra_compute=10,
    num_compute_layers=1
)

msp.data_dir = "../data/chenxi-synthetic-dataset-20210630/synthetic_dataset/canonical_0_False_psam_splice_site_None_200_synthetic_"

msp.extra_params += " --CL_max 400 --data_chunk_to_use all"

msp.acc_thresh = 100

msp.run()
