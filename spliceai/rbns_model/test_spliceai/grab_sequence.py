#!/usr/bin/env python3

import os
from constants import get_args

standard_args = get_args()

CLr = standard_args.CL_max // 2
CLl = CLr + 1
# First nucleotide not included by BEDtools

COMMAND = """cat %s | awk -v CLl=%s -v CLr=%s '{print $3"\t"($5-CLl)"\t"($6+CLr)}' > temp.bed"""

os.system(
    COMMAND
    % (
        standard_args.splice_table,
        CLl,
        CLr,
    )
)

os.system(
    "bedtools getfasta -bed temp.bed -fi {ref_genome} -fo {sequence} -tab".format(
        ref_genome=standard_args.ref_genome, sequence=standard_args.sequence
    )
)

os.system("rm temp.bed")
