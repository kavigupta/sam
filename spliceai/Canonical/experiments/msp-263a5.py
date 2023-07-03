from sys import argv
from msp import MSP

msp = MSP()

msp.batch_size = 2 * msp.batch_size // 3
msp.lr = float(msp.lr) * 2 / 3

msp.file = __file__
msp.seed = int(argv[1])
donor_seed = msp.seed % 1000

msp.architecture["splicepoint_model_spec"]["donor"] = f"model/msp-262da5_{donor_seed}"

msp.architecture["motif_model_spec"]["motif_width"] = 9
msp.architecture["motif_model_spec"]["motif_feature_extractor_spec"] = dict(
    type="FCMotifFeatureExtractor",
    extra_compute=10,
    num_compute_layers=1
)

msp.architecture["num_motifs"] = 4 + 2
msp.n_epochs = 20

msp.acc_thresh = 0
msp.extra_params += " --learned-motif-sparsity-threshold-initial 85"

msp.run()
