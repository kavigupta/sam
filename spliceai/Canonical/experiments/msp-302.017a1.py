from sys import argv
from msp import MSP

from msp_fly import fly_am

msp = MSP()
msp.file = __file__
msp.seed = int(argv[1])
fly_am(msp)

msp.architecture["final_processor_spec"]["bottleneck_spec"] = dict(
    type="ConvolutionalBottleneck", width=17
)

msp.run()
