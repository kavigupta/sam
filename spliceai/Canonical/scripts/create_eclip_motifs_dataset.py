from modular_splicing.eclip.trained_on_eclip.training_data import EclipMotifsDataset

EclipMotifsDataset(
    path="dataset_train_all.h5",
    path_annotation="dataset_intron_exon_annotations_train_all.h5",
    shuffle=True,
    seed=0,
    mode=("from_5'", 50),
)
