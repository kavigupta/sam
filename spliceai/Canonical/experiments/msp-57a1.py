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

assert msp.seed != 1, "shouldn't train seed=1 against seed=1"
msp.extra_params += " --data_dir picf9_pred/"
msp.acc_thresh = 95

msp.run()
