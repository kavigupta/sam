"""
Methods for creating datasets from splice tables.

Puts grab_sequence, create_dataset, and create_datafile together.
"""

from tempfile import NamedTemporaryFile

import h5py

from .create_datafile import create_datafile
from .create_dataset import create_dataset
from .grab_sequence import grab_sequence


def create_dataset_from_splice_table(
    *,
    sequence_path,
    splice_table_frame,
    load_file,
    data_segment_to_use,
    data_chunk_to_use,
    CL_max,
    SL,
    include_seq,
):
    """
    sequence_path: a sequence file created by grab_sequence. Must correspond to the
        splice table (if not an error would be produced)
    splice_table_frame: a dataframe containing the splice table
    load_file: a function to procses the h5py file and produce results
    data_segment_to_use: either "train" or "test"
    data_chunk_to_use: either "all" or a number
    CL_max, SL: max CL to allow and sequence length
    include_seq: whether to include the RNA sequence in the dataset. Set this to False if load_file won't use it
    """
    with NamedTemporaryFile() as f_datatxt, NamedTemporaryFile() as f_datafile, NamedTemporaryFile() as f_dataset:
        splice_table_frame.to_csv(f_datatxt.name, sep="\t", index=False, header=False)
        create_datafile(
            data_segment_to_use=data_segment_to_use,
            data_chunk_to_use=data_chunk_to_use,
            sequence_path=sequence_path,
            splice_table_path=f_datatxt.name,
            CL_max=CL_max,
            outfile=f_datafile.name,
            include_seq=include_seq,
        )
        create_dataset(
            datafile_path=f_datafile.name,
            dataset_path=f_dataset.name,
            CL_max=CL_max,
            SL=SL,
        )
        with h5py.File(f_dataset.name, "r") as f:
            out = load_file(f)
    return out


def create_dataset_from_splice_table_no_sequence(
    *,
    ref_genome,
    splice_table_frame,
    load_file,
    data_segment_chunks_to_use,
    CL_max,
    SL,
    include_seq,
):
    """
    ref_genome: hg19.fa or similar file
    splice_table_frame: a dataframe containing the splice table
    load_file: a function to process the h5py file and produce results
    data_segment_chunks_to_use: a list of tuples of (data_segment, data_chunk)
    CL_max, SL: max CL to allow and sequence length
    include_seq: whether to include the RNA sequence in the dataset. Set this to False if load_file won't use it
    """
    with NamedTemporaryFile() as f_datatxt, NamedTemporaryFile() as sequence_f:
        splice_table_frame.to_csv(f_datatxt.name, sep="\t", index=False, header=False)
        grab_sequence(
            ref_genome=ref_genome,
            splice_table=f_datatxt.name,
            sequence_out=sequence_f.name,
            CL_max=CL_max,
        )

        return {
            (data_segment_to_use, data_chunk_to_use): create_dataset_from_splice_table(
                sequence_path=sequence_f.name,
                splice_table_frame=splice_table_frame,
                load_file=load_file,
                data_segment_to_use=data_segment_to_use,
                data_chunk_to_use=data_chunk_to_use,
                CL_max=CL_max,
                SL=SL,
                include_seq=include_seq,
            )
            for data_segment_to_use, data_chunk_to_use in data_segment_chunks_to_use
        }
