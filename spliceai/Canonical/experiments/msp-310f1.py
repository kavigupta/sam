from sys import argv
from msp import MSP

from msp_fly import fly_am

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])
fly_am(msp)

msp.architecture["propagate_sparsity_spec"] = dict(
    type="OnlyAllowInfluenceOnSplicesites"
)

msp.architecture["final_processor_spec"].update(
    dict(
        bottleneck_spec=dict(type="FirstChannelsBottleneck", num_outputs_kept=2),
        post_bottleneck_lsmrp_spec=dict(type="EarlyChannelLSMRP"),
        evaluate_post_bottleneck=True,
        evaluate_post_bottleneck_weight=0.01,
    )
)
msp.architecture["lsmrp_spec"] = dict(type="DoNotPropagateLSMRP")

msp.architecture["influence_calculator_spec"]["post_sparse_spec"] = dict(
    type="SparsityPropagationPostSparse",
    underlying_spec=msp.architecture["influence_calculator_spec"]["post_sparse_spec"],
    sparsity_propagation_spec=dict(type="AddOnlyToNonZero"),
)

msp.run()
