###############################################################################
"""This parser takes as input the text files canonical_dataset.txt and
canonical_sequence.txt, and produces a .h5 file datafile_{}_{}.h5,
which will be later processed to create dataset_{}_{}.h5. The file
dataset_{}_{}.h5 will have datapoints of the form (X,Y), and can be
understood by models."""
###############################################################################

import numpy as np
import re
import sys
import time
import h5py
from constants import get_args

standard_args = get_args()

start_time = time.time()

assert standard_args.data_segment_to_use in ["train", "test", "all"]
assert standard_args.data_chunk_to_use in ["0", "1", "all"]

if standard_args.data_segment_to_use == "train":
    CHROM_GROUP = [
        "chr11",
        "chr13",
        "chr15",
        "chr17",
        "chr19",
        "chr21",
        "chr2",
        "chr4",
        "chr6",
        "chr8",
        "chr10",
        "chr12",
        "chr14",
        "chr16",
        "chr18",
        "chr20",
        "chr22",
        "chrX",
        "chrY",
    ]
elif standard_args.data_segment_to_use == "test":
    CHROM_GROUP = ["chr1", "chr3", "chr5", "chr7", "chr9"]
else:
    CHROM_GROUP = [
        "chr1",
        "chr3",
        "chr5",
        "chr7",
        "chr9",
        "chr11",
        "chr13",
        "chr15",
        "chr17",
        "chr19",
        "chr21",
        "chr2",
        "chr4",
        "chr6",
        "chr8",
        "chr10",
        "chr12",
        "chr14",
        "chr16",
        "chr18",
        "chr20",
        "chr22",
        "chrX",
        "chrY",
    ]

###############################################################################

NAME = []  # Gene symbol
PARALOG = []  # 0 if no paralogs exist, 1 otherwise
CHROM = []  # Chromosome number
STRAND = []  # Strand in which the gene lies (+ or -)
TX_START = []  # Position where transcription starts
TX_END = []  # Position where transcription ends
JN_START = []  # Positions where canonical exons end
JN_END = []  # Positions where canonical exons start
SEQ = []  # Nucleotide sequence

fpr2 = open(standard_args.sequence, "rb")

with open(standard_args.splice_table, "rb") as fpr1:
    for line1 in fpr1: # 

        line2 = fpr2.readline()

        data1 = re.split(b"\n|\t", line1)[:-1]
        data2 = re.split(b"\n|\t|:|-", line2)[:-1]

        assert data1[2] == data2[0]
        assert int(data1[4]) == int(data2[1]) + standard_args.CL_max // 2 + 1
        assert int(data1[5]) == int(data2[2]) - standard_args.CL_max // 2

        if data1[2].decode("utf-8") not in CHROM_GROUP:
            continue

        if (standard_args.data_chunk_to_use != data1[1].decode("utf-8")) and (
            standard_args.data_chunk_to_use != "all"
        ):
            continue

        NAME.append(data1[0])
        PARALOG.append(int(data1[1]))
        CHROM.append(data1[2])
        STRAND.append(data1[3])
        TX_START.append(data1[4])
        TX_END.append(data1[5])
        JN_START.append(data1[6::2])
        JN_END.append(data1[7::2])
        SEQ.append(data2[3]) # the sequence

fpr1.close()
fpr2.close()

###############################################################################

h5f = h5py.File(
    standard_args.data_dir
    + "datafile"
    + "_"
    + standard_args.data_segment_to_use
    + "_"
    + standard_args.data_chunk_to_use
    + ".h5",
    "w",
)

h5f.create_dataset("NAME", data=np.asarray(NAME))
h5f.create_dataset("PARALOG", data=np.asarray(PARALOG))
h5f.create_dataset("CHROM", data=np.asarray(CHROM))
h5f.create_dataset("STRAND", data=np.asarray(STRAND))
h5f.create_dataset("TX_START", data=np.asarray(TX_START))
h5f.create_dataset("TX_END", data=np.asarray(TX_END))
h5f.create_dataset("JN_START", data=np.asarray(JN_START))
h5f.create_dataset("JN_END", data=np.asarray(JN_END))
h5f.create_dataset("SEQ", data=np.asarray(SEQ))

h5f.close()

print("--- %s seconds ---" % (time.time() - start_time))

###############################################################################
