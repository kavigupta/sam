from sys import argv
from msp import MSP, step_for_density

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])

msp.lr = float(msp.lr) * 0.9 ** 16
path = f"model/rbnsp-80-adj2_{msp.seed}"
msp.architecture = dict(
    type="DroppedMotifFromModel",
    original_model_path=path,
    original_model_step=step_for_density(path, 0.178e-2),
    dropped_motifs=[],
    finetune_motifs=True,
)

msp.train_spec = dict(
    type="single_perturbed",
    epochs_per_motif=0.5,
    num_motifs=79,
)
msp.run()
