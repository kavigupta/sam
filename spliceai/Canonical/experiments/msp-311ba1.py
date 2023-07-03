from sys import argv
from msp import MSP
from msp_donorlike import setup_as_adjusted_donor, train_in_parallel

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])

setup_as_adjusted_donor(msp)


msp.architecture["sparse_spec"] = dict(
    type="ParallelSpatiallySparse",
    sparse_specs=[
        dict(type="NoSparsity"),
        msp.architecture["sparse_spec"],
    ],
    num_channels_each=[1, 79],
    update_indices=[1],
    get_index=1,
)

train_in_parallel(msp)


msp.run()
