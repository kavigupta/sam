from sys import argv
from msp import MSP

msp = MSP()

msp.batch_size = 2 * msp.batch_size // 3
msp.lr = float(msp.lr) * 2 / 3

msp.file = __file__
msp.seed = int(argv[1])
msp.architecture["motif_model_spec"]["motif_width"] = 9
msp.architecture["motif_model_spec"]["motif_feature_extractor_spec"] = dict(
    type="FCMotifFeatureExtractor",
    extra_compute=10,
    num_compute_layers=1
)
msp.architecture["num_motifs"] = 3 + 2
msp.run()
