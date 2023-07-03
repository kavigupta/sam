from sys import argv
from msp import MSP, rbnsp_split_90

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])

msp.batch_size = msp.batch_size // 2
msp.lr = float(msp.lr) * 1 / 2

num_lm = 10
num_psams = 80
msp.architecture["num_motifs"] = num_lm + num_psams + 2
msp.architecture["channels"] = 200

psam_model = dict(
    type="PretrainedMotifModel",
    model_path="model/msp-134ab1_1",
    model_step=2753370,
    finetunable=False,
)
lm = msp.architecture["motif_model_spec"]
lm["motif_width"] = 9
lm["motif_feature_extractor_spec"] = dict(
    type="FCMotifFeatureExtractor", extra_compute=10, num_compute_layers=1
)

msp.architecture["motif_model_spec"] = dict(
    type="ParallelMotifModels",
    specs=[psam_model, lm],
    num_motifs_each=[num_psams, num_lm],
    notify_sparsity_idxs=[1],
)

msp.architecture["sparse_spec"] = dict(
    type="ParallelSpatiallySparse",
    sparse_specs=[
        dict(type="SpatiallySparseAcrossChannels", sparsity=0.5),
        dict(type="SpatiallySparseAcrossChannels", sparsity=0.5),
    ],
    num_channels_each=[num_psams, num_lm],
    update_indices=[1],
    get_index=1,
)

msp.acc_thresh = 82

msp.run()
