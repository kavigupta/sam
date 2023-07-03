from modular_splicing.dataset.h5_dataset import H5Dataset


def basic_dataset(
    path,
    cl,
    cl_max,
    *,
    iterator_spec=dict(type="FastIter", shuffler_spec=dict(type="DoNotShuffle")),
    post_processor_spec=dict(
        type="FlattenerPostProcessor", indices=[("inputs", "x"), ("outputs", "y")]
    ),
    **kwargs
):
    """
    Simple method to automatically create a dataset. Can be customized.
    """
    return H5Dataset(
        path=path,
        cl=cl,
        cl_max=cl_max,
        iterator_spec=iterator_spec,
        **kwargs,
        datapoint_extractor_spec=dict(type="BasicDatapointExtractor"),
        post_processor_spec=post_processor_spec,
    )
