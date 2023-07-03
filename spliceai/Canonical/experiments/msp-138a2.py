from sys import argv
from msp import MSP

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])
msp.architecture["num_motifs"] = 80
msp.architecture["motif_model_spec"] = dict(
    type="NeuralFixedMotif",
    path_format="model/rbns-binary-model-{motif}-2",
    removed_motifs=["3P", "5P", "HNRNPA1", "RALY"],
)
msp.run()
