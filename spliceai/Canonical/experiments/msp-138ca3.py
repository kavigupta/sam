from sys import argv
from msp import MSP

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])
msp.architecture["num_motifs"] = 80
msp.architecture["sparse_spec"] = dict(type="SpatiallySparseAcrossChannels")

msp.architecture["motif_model_spec"] = dict(
    type="NeuralFixedMotif",
    path_format="model/rbns-binary-model-{motif}-13x4",
    removed_motifs=["3P", "5P", "HNRNPA1", "RALY"],
)

msp.architecture["affine_sparsity_enforcer"] = True
msp.architecture["propagate_sparsity_spec"] = dict(type="NoSparsityPropagation")

msp.acc_thresh = 0
msp.extra_params += " --learned-motif-sparsity-threshold-initial 85"

msp.n_epochs = 40

msp.run()