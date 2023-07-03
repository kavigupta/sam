from sys import argv
from msp import MSP

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])
msp.architecture["motif_model_spec"]["motif_width"] = 13
msp.architecture["motif_model_spec"]["motif_feature_extractor_spec"] = dict(
    type="FCMotifFeatureExtractor",
    extra_compute=10,
    num_compute_layers=1
)
msp.extra_params += " --learned-motif-sparsity-drop-motif-frequency 0.87"
msp.extra_params += " --learned-motif-sparsity-threshold-initial 90"
msp.run()
