from sys import argv
from msp import MSP, rbnsp_split_90

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])
msp.architecture["num_motifs"] = 172
msp.architecture["channels"] = 200

psam_model = dict(
    type="PSAMMotifModel",
    motif_spec=dict(type="rbns"),
    exclude_names=("3P", "5P") + tuple(rbnsp_split_90["exclude"]),
    rbns_psam_spec=dict(type="RbnsPsamMotifs", thresholding_mode="subtractive"),
)
lm = msp.architecture["motif_model_spec"]
lm["motif_width"] = 9
lm["motif_feature_extractor_spec"] = dict(
    type="FCMotifFeatureExtractor", extra_compute=10, num_compute_layers=1
)

msp.architecture["motif_model_spec"] = dict(
    type="ParallelMotifModels",
    specs=[psam_model, lm],
    num_motifs_each=[72, 98],
)

msp.architecture["sparse_spec"] = dict(
    type="ParallelSpatiallySparse",
    sparse_specs=[
        dict(
            type="SpatiallySparseAcrossChannels",
            sparsity=0
        ),
        dict(type="SpatiallySparseAcrossChannels"),
    ],
    num_channels_each=[72, 98],
    update_indices=[1],
    get_index=1,
)

msp.run()
