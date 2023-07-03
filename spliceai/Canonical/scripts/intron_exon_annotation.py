import tqdm.auto as tqdm
import h5py

from modular_splicing.utils.intron_exon_annotation import (
    to_onehot_intron_exon_annotation,
)


def add_annotations(input_file, output_file):
    with h5py.File(input_file, "r") as f:
        amount = len(f) // 2
        with h5py.File(output_file, "w") as g:
            for idx in tqdm.trange(amount):
                g.create_dataset(f"X{idx}", data=f[f"X{idx}"])
                g.create_dataset(
                    f"Y{idx}", data=to_onehot_intron_exon_annotation(f[f"Y{idx}"])
                )


def main():
    add_annotations("dataset_test_0.h5", "dataset_intron_exon_annotations_test_0.h5")
    add_annotations(
        "dataset_train_all.h5", "dataset_intron_exon_annotations_train_all.h5"
    )


if __name__ == "__main__":
    main()
