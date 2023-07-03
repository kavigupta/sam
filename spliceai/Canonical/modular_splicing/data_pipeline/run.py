import os
import fire

from .grab_sequence import grab_sequence
from .create_datafile import create_datafile
from .create_dataset import create_dataset


def full_pipeline(
    *,
    ref_genome,
    splice_table,
    sequence_out,
    data_segments,
    data_dir,
    CL_max=10_000,
    SL=5000,
):
    grab_sequence(
        ref_genome=ref_genome,
        splice_table=splice_table,
        sequence_out=sequence_out,
        CL_max=CL_max,
    )
    for segment in data_segments:
        chunk = {"all": "all", "train": "all", "test": "0"}[segment]

        data_path = os.path.join(data_dir, f"%s_{segment}_{chunk}.h5")

        create_datafile(
            data_segment_to_use=segment,
            data_chunk_to_use=chunk,
            sequence_path=sequence_out,
            splice_table_path=splice_table,
            CL_max=CL_max,
            outfile=data_path % "datafile",
            include_seq=True,
        )
        create_dataset(
            datafile_path=data_path % "datafile",
            dataset_path=data_path % "dataset",
            SL=SL,
            CL_max=CL_max,
        )


def canonical_dataset(ref_genome):
    return full_pipeline(
        ref_genome=ref_genome,
        splice_table="canonical_dataset.txt",
        sequence_out="canonical_sequence.txt",
        data_segments=["train", "test"],
        data_dir="./",
        CL_max=10_000,
        SL=5000,
    )


def evo_dataset(ref_genome):
    return full_pipeline(
        ref_genome=ref_genome,
        splice_table="../data/evo_alt_data/evo_all.txt",
        sequence_out="../data/evo_alt_data/evo/sequence.txt",
        data_segments=["all"],
        data_dir="../data/evo_alt_data/evo/",
        CL_max=10_000,
        SL=5000,
    )


def alt_dataset(ref_genome):
    return full_pipeline(
        ref_genome=ref_genome,
        splice_table="../data/evo_alt_data/alt_all.txt",
        sequence_out="../data/evo_alt_data/alt/sequence.txt",
        data_segments=["all"],
        data_dir="../data/evo_alt_data/alt/",
        CL_max=10_000,
        SL=5000,
    )


def evo_alt_dataset(ref_genome):
    evo_dataset(ref_genome)
    alt_dataset(ref_genome)


if __name__ == "__main__":
    fire.Fire()
