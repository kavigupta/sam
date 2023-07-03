import os
import tempfile
import numpy as np
import re
import h5py
import tqdm.auto as tqdm

from modular_splicing.data_pipeline.grab_sequence import grab_sequence


def create_datafile(
    *,
    data_segment_to_use,
    data_chunk_to_use,
    sequence_path,
    splice_table_path,
    CL_max,
    outfile,
    include_seq=True,
):
    """
    Create a datafile from a splice table and a sequence file.

    data_segment_to_use: the data segment to use, either "train" or "test"
    data_chunk_to_use: the data chunk to use, either "all" or 0, or 1. Use 0 to disable paralogs.
    sequence_path: the path to the sequence file produced by grab_sequence
    splice_table_path: the path to the splice table file
    CL_max: the maximum CL to allow later on
    outfile: the path to the output file
    include_seq: whether to include the RNA sequence in the dataset. Set this to False if you just want the outputs
    """
    assert data_segment_to_use in ["train", "test", "all"]
    assert data_chunk_to_use in ["0", "1", "all"]

    if data_segment_to_use == "train":
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
    elif data_segment_to_use == "test":
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

    fpr2 = open(sequence_path, "rb")

    with open(splice_table_path, "rb") as fpr1:
        for line1 in tqdm.tqdm(list(fpr1)):
            line2 = fpr2.readline()

            data1 = re.split(b"\n|\t", line1)[:-1]
            data2 = re.split(b"\n|\t|:|-", line2)[:-1]

            assert data1[2] == data2[0]
            assert int(data1[4]) == int(data2[1]) + CL_max // 2 + 1
            assert int(data1[5]) == int(data2[2]) - CL_max // 2

            if data1[2].decode("utf-8") not in CHROM_GROUP:
                continue

            if (data_chunk_to_use != data1[1].decode("utf-8")) and (
                data_chunk_to_use != "all"
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
            if include_seq:
                SEQ.append(data2[3])

    fpr1.close()
    fpr2.close()

    ###############################################################################

    h5f = h5py.File(outfile, "w")

    h5f.create_dataset("NAME", data=np.asarray(NAME))
    h5f.create_dataset("PARALOG", data=np.asarray(PARALOG))
    h5f.create_dataset("CHROM", data=np.asarray(CHROM))
    h5f.create_dataset("STRAND", data=np.asarray(STRAND))
    h5f.create_dataset("TX_START", data=np.asarray(TX_START))
    h5f.create_dataset("TX_END", data=np.asarray(TX_END))
    h5f.create_dataset("JN_START", data=np.asarray(JN_START))
    h5f.create_dataset("JN_END", data=np.asarray(JN_END))
    if include_seq:
        h5f.create_dataset("SEQ", data=np.asarray(SEQ))

    h5f.close()

    ###############################################################################


def produce_datafiles(
    splice_table, *, fasta_path, data_path_folder, CL_max, segment_chunks
):
    """
    Produce datafiles for each segment and chunk (paralog class).

    Parameters
    ----------
    splice_table : pd.DataFrame
        Splice table with columns `name`, `chr`, `strand`, `start`, `end`, `chunk`.
    fasta_path : str
        Path to reference genome fasta file.
    data_path_folder : str
        Path to folder where datafiles will be saved.
    CL_max : int
        Maximum number of chunks to use.
    segment_chunks : list of tuples
        List of tuples of segment and chunk to use.
    """

    outfiles = {
        (segment, chunk): f"{data_path_folder}datafile_{segment}_{chunk}.h5"
        for segment, chunk in segment_chunks
    }

    if all(os.path.exists(outfile) for outfile in outfiles.values()):
        print("Datafiles already exist, skipping creation.")
        return

    with tempfile.NamedTemporaryFile() as splice_table_f, tempfile.NamedTemporaryFile() as genes_sequence_f:
        splice_table.to_csv(splice_table_f.name, index=False, header=False, sep="\t")
        grab_sequence(
            ref_genome=fasta_path,
            sequence_out=genes_sequence_f.name,
            splice_table=splice_table_f.name,
            CL_max=CL_max,
        )
        for segment, chunk in segment_chunks:
            create_datafile(
                data_segment_to_use=segment,
                data_chunk_to_use=chunk,
                sequence_path=genes_sequence_f.name,
                splice_table_path=splice_table_f.name,
                CL_max=CL_max,
                outfile=outfiles[segment, chunk],
            )
