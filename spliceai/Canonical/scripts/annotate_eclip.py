from modular_splicing.eclip.data.pipeline import produce_all_eclips

RCS = "1", "2", "1, 2"
for rc in RCS:
    for is_train in False, True:
        produce_all_eclips(
            replicate_category=rc,
            dataset_path="canonical_dataset.txt",
            sequence_path="canonical_sequence.txt",
            is_train=is_train,
        )
