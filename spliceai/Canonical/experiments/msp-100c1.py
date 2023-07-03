from sys import argv
from msp import MSP

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])
msp.architecture["motif_model_spec"]["motif_width"] = 9
msp.architecture["motif_model_spec"]["motif_feature_extractor_spec"] = dict(
    type="FCMotifFeatureExtractor", extra_compute=10, num_compute_layers=1
)
msp.architecture["motif_model_spec"]["motif_feature_extractor_spec"] = dict(
    type="FCMotifFeatureExtractor", extra_compute=10, num_compute_layers=1
)

msp.architecture["influence_calculator_spec"]["long_range_reprocessor_spec"][
    "forward_only"
] = True

msp.architecture["final_processor_spec"]["long_range_final_layer_spec"] = dict(
    type="LSTMLongRangeFinalLayer", forward_only=True
)

msp.acc_thresh = 70

msp.run()
