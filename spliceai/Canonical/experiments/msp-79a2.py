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
msp.architecture["influence_calculator_spec"]["long_range_reprocessor_spec"]["max_len"] = 20_000

msp.SL = 15_000
msp.window = 1200
msp.batch_size //= 3

msp.extra_params += " --data_dir ../data/sl-15k/ "

msp.run()
