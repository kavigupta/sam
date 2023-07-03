import os
from modular_splicing.data_pipeline.create_dataset import create_dataset

datafiles_from_chenxi = {
    "fly": "../data/drosophila_model_and_data/drosophila_datafile_test_0.h5",
    "mouse": "../data/mouse_model_and_data/musculus_datafile_test_0.h5",
}


def dataset_for_species(species):
    """
    Provide the dataset for the given species.

    Parameters
    ----------
    species : str
        The species to provide the dataset for.

    Returns (datafile, dataset, kwargs)
    -------
    datafile : str
        The path to the datafile.
    dataset : str
        The path to the dataset.
    kwargs : dict
        The kwargs to pass to the dataset constructor.
    """
    if species == "human":
        return "datafile_test_0.h5", "dataset_test_0.h5", dict()
    elif species in datafiles_from_chenxi:
        datafile = datafiles_from_chenxi[species]
        dataset = datafile.replace("datafile", "dataset")
        if not os.path.exists(dataset):
            create_dataset(
                datafile_path=datafile, dataset_path=dataset, SL=5000, CL_max=400
            )
        return datafile, dataset, dict(cl_max=400)
    raise ValueError(f"Unknown species {species}")
