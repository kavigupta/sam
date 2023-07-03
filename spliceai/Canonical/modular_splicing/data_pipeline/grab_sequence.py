#!/usr/bin/env python3

import subprocess
import os
import tempfile


def grab_sequence(*, ref_genome, splice_table, sequence_out, CL_max):
    _, temp_bed = tempfile.mkstemp(suffix=".bed")

    CLr = CL_max // 2
    CLl = CLr + 1
    # First nucleotide not included by BEDtools

    COMMAND = (
        """cat %s | awk -v CLl=%s -v CLr=%s '{print $3"\t"($5-CLl)"\t"($6+CLr)}' > %s"""
    )

    subprocess.check_call(COMMAND % (splice_table, CLl, CLr, temp_bed), shell=True)

    subprocess.check_call(
        f"bedtools getfasta -bed {temp_bed} -fi {ref_genome} -fo {sequence_out} -tab",
        shell=True,
    )

    os.unlink(temp_bed)
