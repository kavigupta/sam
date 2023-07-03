from sys import argv
from msp import MSP

from msp_fly import fly_am

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])
fly_am(msp)

msp.architecture["final_processor_spec"].update(
    dict(
        bottleneck_spec=dict(type="ConvolutionalBottleneck", width=65),
        post_bottleneck_lsmrp_spec=dict(type="EarlyChannelLSMRP"),
        evaluate_post_bottleneck=True,
        evaluate_post_bottleneck_weight=0.01,
    )
)
msp.architecture["lsmrp_spec"] = dict(type="DoNotPropagateLSMRP")

msp.run()
