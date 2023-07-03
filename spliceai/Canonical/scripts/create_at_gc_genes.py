from modular_splicing.data_pipeline.create_dataset import create_dataset

THRESHOLD_MEDIAN_AT = 0.5416523608512839

for folder, filter in [
    ("at-rich", lambda x: x > THRESHOLD_MEDIAN_AT),
    ("gc-rich", lambda x: x <= THRESHOLD_MEDIAN_AT),
]:
    for suffix in "test_0", "train_all":
        print(folder, suffix)
        create_dataset(
            datafile_path=f"datafile_{suffix}.h5",
            dataset_path=f"../data/by-at-richness/{folder}/dataset_{suffix}.h5",
            SL=5000,
            CL_max=10_000,
            at_richness_filter=filter,
        )
