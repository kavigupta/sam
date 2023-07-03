from sys import argv
from msp import MSP

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])
msp.architecture["num_motifs"] = 80 + 2
msp.architecture["channels"] = 200
msp.architecture["motif_model_spec"] = dict(type="NoMotifModel")

msp.architecture["propagate_sparsity_spec"] = dict(type="NoSparsityPropagation")
msp.data_dir = "../data/chenxi-synthetic-dataset-20211210-tra2a/synthetic_dataset_TRA2A/canonical_1_False_fixed_more_top_200_TRA2A_synthetic_"

msp.extra_params += " --CL_max 2000 --data_chunk_to_use all"

msp.acc_thresh = 100

msp.run()
