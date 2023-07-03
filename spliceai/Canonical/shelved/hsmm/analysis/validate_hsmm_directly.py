import numpy as np


from permacache import permacache, stable_hash

from ..direct_inference_experiment import directly_infer_output


@permacache("hsmm/analysis/validate_hsmm_directly", key_function=dict(sm=stable_hash))
def compute_hsmm(hsmm_spec, sm, config, limit, transformation_spec):
    b_s, a_s, y_s = [], [], []
    for x, y, _ in config.iterate_data(use="tune_indices", limit=limit):
        b, a = directly_infer_output(
            hsmm_spec,
            sm,
            x[:10000],
            transformation_spec=transformation_spec,
        )
        b_s.append(b)
        a_s.append(a)
        y_s.append(y[:10_000, 1:])
    b_s = np.concatenate(b_s, axis=1).T
    a_s = np.concatenate(a_s, axis=1).T
    y_s = np.concatenate(y_s)
    o = acc(y_s, b_s).mean()
    m = acc(y_s, a_s).mean()
    print(f"Original: {o:.2%}")
    print(f"Modified: {m:.2%}")
    return dict(acc_o=o, acc_m=m, b_s=b_s, a_s=a_s, y_s=y_s)


def acc(y_s, yp_s):
    thresholds = [np.quantile(yp_s[:, c], 1 - y_s[:, c].mean()) for c in range(2)]
    return ((yp_s > thresholds) & y_s).sum(0) / y_s.sum(0)
