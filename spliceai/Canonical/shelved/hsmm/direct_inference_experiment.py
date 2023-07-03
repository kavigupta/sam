from permacache import permacache, stable_hash

import torch

from modular_splicing.utils.construct import construct

from .kayla_gene_model import all_transformed_hsmm
from .transformed_models import unwrap_state


@permacache(
    "notebooks/hsmm/direct_inference_experiment/directly_infer_output_5",
    key_function=dict(sm=stable_hash, x=stable_hash),
)
def directly_infer_output(
    hsmm_t_spec, sm, x, transformation_spec=dict(type="Identity")
):
    infer = (
        lambda hsmm_t, backmap, overall: hsmm_t.hsmm.to_torch(trainable=False)
        .cuda()
        .infer_states(overall[:, :, :])
        .cpu()
        .numpy()[0, get_splicepoint_idxs(backmap)]
    )

    return directly_infer_output_generically(
        hsmm_t_spec, sm, x, transformation_spec=transformation_spec, infer=infer
    )


def directly_infer_output_generically(
    hsmm_t_spec, sm, x, *, transformation_spec, infer
):
    hsmm_t = construct(all_transformed_hsmm(), hsmm_t_spec)

    backmap = [unwrap_state(x) for x in hsmm_t.hsmm.states]

    with torch.no_grad():
        splicepoints = compute_splicepoints(sm, x)
        original_splicepoints = splicepoints.T
        splicepoints = torch.tensor([splicepoints]).cuda()

        splicepoints = construct(
            dict(
                Identity=lambda splicepoints: splicepoints,
                Scale=lambda splicepoints, factor: splicepoints * factor,
                Add=lambda splicepoints, factor: splicepoints + factor,
            ),
            transformation_spec,
            splicepoints=splicepoints,
        )
        overall = full_emission_probs(splicepoints, backmap)

        return (
            original_splicepoints,
            infer(hsmm_t, backmap, overall),
        )


def get_splicepoint_idxs(backmap):
    splicepoint_idxs = [backmap.index("3'"), backmap.index("5'")]
    return splicepoint_idxs


def full_emission_probs(splicepoints, backmap):
    other = torch.log1p(-splicepoints.exp().sum(-1)) - 1
    overall = other[:, None].repeat(1, len(backmap), 1)
    overall[:, get_splicepoint_idxs(backmap)] = splicepoints.transpose(1, 2)
    overall[:, backmap.index("done")] = -300
    return overall


@permacache(
    "hsmm/direct_inference_experiment/compute_splicepoints_2",
    key_function=dict(sm=stable_hash, x=stable_hash),
)
def compute_splicepoints(sm, x):
    splicepoints = sm.forward_just_splicepoints(torch.tensor([x]).float().cuda())[0]
    return splicepoints.cpu().numpy()
