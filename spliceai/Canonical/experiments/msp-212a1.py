from sys import argv
from msp import MSP

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])

msp.lr = float(msp.lr) * 0.9 ** 16
msp.architecture = dict(
    type="DroppedMotifFromModel",
    original_model_path="model/msp-173a1_1",
    original_model_step=1301680,
    dropped_motifs=[],
)

msp.train_spec = dict(
    type="auto_minimizer",
    train_steps=20_000,
    train_select_steps=5_000,
    test_steps=5_000,
    trainer_spec=dict(
        type="MotifIncrementalDropper",
        names_spec=dict(type="rbnsp")
    ),
)
msp.run()
