from sys import argv
from msp import MSP

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])

msp.architecture["splicepoint_model_spec"] = dict(
    type="BothLSSIModels",
    acceptor="model/splicepoint-model-acceptor-1",
    donor="model/splicepoint-model-donor-1",
)

am_e = dict(
    type="PretrainedEclipMotifModel",
    stub_spec=dict(
        type="model_trained_on_eclip",
        motif_names_source="eclip_18",
        eclip_model_spec="am_21x2_178_post_sparse_scale",
        seed=1,
    ),
    set_eclip_models_to_eval=True,
)

msp.architecture["motif_model_spec"] = dict(
    type="ParallelMotifModels",
    specs=[am_e, dict(type="NoMotifModel")],
    num_motifs_each=[18, 62],
)


msp.architecture["num_motifs"] = 82
msp.architecture["sparse_spec"] = dict(type="SpatiallySparseAcrossChannels")
msp.architecture["affine_sparsity_enforcer"] = True

msp.acc_thresh = 100

msp.n_epochs = 40

msp.run()
