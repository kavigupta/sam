from functools import lru_cache

import numpy as np

from modular_splicing.evaluation import standard_e2e_eval


@lru_cache(None)
def second_half_of_spliceai_cached():
    data_spec = dict(
        type="H5Dataset",
        sl=5000,
        datapoint_extractor_spec=dict(type="BasicDatapointExtractor"),
        post_processor_spec=dict(type="IdentityPostProcessor"),
    )
    items = list(standard_e2e_eval.test_data(data_spec, cl=50))
    xs, ys = [item["inputs"]["x"] for item in items], [
        item["outputs"]["y"] for item in items
    ]
    xs, ys = np.array(xs), np.array(ys)
    return xs, ys
