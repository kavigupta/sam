from sys import argv
from msp import MSP

CL_VALUES = list(range(0, 1 + 20, 2))

for splicepoint_type in "donor", "acceptor":
    for left in CL_VALUES:
        for right in CL_VALUES:
            print(f"{splicepoint_type} {left} {right}")
            msp = MSP()
            msp.file = __file__
            msp.seed = int(argv[1])
            msp.lr = 1e-3
            msp.window = 40
            msp.batch_size = 150
            msp.architecture = dict(
                type="SplicePointIdentifier",
                cl=40,
                asymmetric_cl=(left, right),
                hidden_size=100,
                n_layers=4,
            )

            msp.extra_params += f" --only-train {splicepoint_type}"

            msp.n_epochs = 10

            msp.extra_name_for_path = f"-{splicepoint_type},{left},{right}"

            msp.run()
