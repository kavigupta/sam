import matplotlib.pyplot as plt
import pandas as pd

from modular_splicing.utils.plots.results_by_model_group import plot_grouped_results
from modular_splicing.evaluation.evaluate_model_series import (
    evaluate_all_series,
)

from modular_splicing.models_for_testing.models_for_testing import (
    EndToEndModelsForTesting,
    STANDARD_DENSITY,
    STANDARD_DENSITY_FLY,
)

normal_spec = dict(
    type="H5Dataset",
    sl=5000,
    datapoint_extractor_spec=dict(
        type="BasicDatapointExtractor",
    ),
    post_processor_spec=dict(type="IdentityPostProcessor"),
)

DATA_PATH_BY_SPECIES = {
    "human": "dataset_test_0.h5",
    "fly": "../rbns_model/organism/drosophila/10k_updated/drosophila_dataset_test_0.h5",
}


def plot_aggs(aggregators, colors, species="human"):
    res = evaluate_all_series(
        *[(ms, normal_spec) for ms in aggregators],
        split="val",
        include_path=True,
        data_path=DATA_PATH_BY_SPECIES[species],
    )
    plt.figure(dpi=200, facecolor="white")
    plot_grouped_results(res, colors=colors(res), ax=plt.gca())
    return pd.DataFrame(res).T


am = [
    EndToEndModelsForTesting(
        name="AM",
        path_prefix="model/msp-265a1",
        seeds=(1, 2, 3),
        density=STANDARD_DENSITY,
    )
]

multi_lengths = [
    EndToEndModelsForTesting(
        name="AM + btl(1)",
        path_prefix="model/msp-292s1",
        seeds=(1, 2, 3),
        density=STANDARD_DENSITY,
    ),
    EndToEndModelsForTesting(
        name="AM + btl(5)",
        path_prefix="model/msp-292w1",
        seeds=(1, 2, 3),
        density=STANDARD_DENSITY,
    ),
    EndToEndModelsForTesting(
        name="AM + btl(9)",
        path_prefix="model/msp-292x1",
        seeds=(1, 2, 3),
        density=STANDARD_DENSITY,
    ),
    EndToEndModelsForTesting(
        name="AM + btl(17)",
        path_prefix="model/msp-292y1",
        seeds=(1, 2, 3),
        density=STANDARD_DENSITY,
    ),
    EndToEndModelsForTesting(
        name="AM + btl(33)",
        path_prefix="model/msp-292z1",
        seeds=(1, 2, 3),
        density=STANDARD_DENSITY,
    ),
    EndToEndModelsForTesting(
        name="AM + btl(65)",
        path_prefix="model/msp-292a1",
        seeds=(1, 2, 3),
        density=STANDARD_DENSITY,
    ),
    EndToEndModelsForTesting(
        name="AM + btl(129)",
        path_prefix="model/msp-292b1",
        seeds=(1, 2, 3),
        density=STANDARD_DENSITY,
    ),
    EndToEndModelsForTesting(
        name="AM + btl(257)",
        path_prefix="model/msp-292c1",
        seeds=(1, 2, 3),
        density=STANDARD_DENSITY,
    ),
]

bottleneck_baseline = [
    mod
    for mod in multi_lengths
    if any(x in mod.name for x in ("btl(17)", "btl(65)", "btl(257)"))
]


bottleneck = [
    EndToEndModelsForTesting(
        name="AM + btl(17) + pre",
        path_prefix="model/msp-293x1",
        seeds=(1, 2, 3),
        density=STANDARD_DENSITY,
    ),
    EndToEndModelsForTesting(
        name="AM + btl(65) + pre",
        path_prefix="model/msp-293a1",
        seeds=(1, 2, 3),
        density=STANDARD_DENSITY,
    ),
    EndToEndModelsForTesting(
        name="AM + btl(257) + pre",
        path_prefix="model/msp-293c1",
        seeds=(1, 2, 3),
        density=STANDARD_DENSITY,
    ),
    EndToEndModelsForTesting(
        name="AM + btl(17) + pre + loss",
        path_prefix="model/msp-294x2",
        seeds=(1, 2, 3),
        density=STANDARD_DENSITY,
    ),
    EndToEndModelsForTesting(
        name="AM + btl(65) + pre + loss",
        path_prefix="model/msp-294a2",
        seeds=(1, 2, 3),
        density=STANDARD_DENSITY,
    ),
    EndToEndModelsForTesting(
        name="AM + btl(257) + pre + loss",
        path_prefix="model/msp-294c2",
        seeds=(1, 2, 3),
        density=STANDARD_DENSITY,
    ),
]

influence_baseline = [
    mod for mod in bottleneck if any(x in mod.name for x in ("btl(17) + pre + loss",))
]

influence = [
    EndToEndModelsForTesting(
        name="AM + btl(17) + pre + loss + FTLE(1)",
        path_prefix="model/msp-298aa1",
        seeds=(1, 2, 3),
        density=STANDARD_DENSITY,
    ),
    EndToEndModelsForTesting(
        name="AM + btl(17) + pre + loss + FTLE(4)",
        path_prefix="model/msp-298ac.a2",
        seeds=(1, 2, 3),
        density=STANDARD_DENSITY,
    ),
    EndToEndModelsForTesting(
        name="AM + btl(17) + pre + loss + FTLE(16)",
        path_prefix="model/msp-298ae.a2",
        seeds=(1, 2, 3),
        density=STANDARD_DENSITY,
    ),
    EndToEndModelsForTesting(
        name="AM + btl(17) + pre + loss + LE(8)LE(4)",
        path_prefix="model/msp-299cc.a1",
        seeds=(1, 2, 3),
        density=STANDARD_DENSITY,
    ),
    EndToEndModelsForTesting(
        name="AM + btl(17) + pre + loss + SG(4)LE(4)",
        path_prefix="model/msp-300bc1",
        seeds=(1, 2),
        density=STANDARD_DENSITY,
    ),
]

# FLY

am_fly = [
    EndToEndModelsForTesting(
        name="AM [fly]",
        path_prefix="model/msp-301a1",
        seeds=(1, 2, 3),
        density=STANDARD_DENSITY_FLY,
    )
]

bottlenecks_fly = [
    EndToEndModelsForTesting(
        name=f"AM [fly] btl({btl})",
        path_prefix=f"model/msp-302.{btl:03d}a1",
        seeds=(1,),
        density=STANDARD_DENSITY_FLY,
    )
    for btl in (5, 17, 65)
]

bottlenecks_prelsmrp_fly = [
    EndToEndModelsForTesting(
        name=f"AM [fly] btl({btl}) + pre",
        path_prefix=f"model/msp-303.{btl:03d}a1",
        seeds=(1,),
        density=STANDARD_DENSITY_FLY,
    )
    for btl in (5, 17, 65)
]

bottlenecks_with_loss_fly = [
    EndToEndModelsForTesting(
        name=f"AM [fly] btl({btl}) + pre + {loss_amount}",
        path_prefix=f"model/msp-304.{btl:03d}{letter}",
        seeds=tuple(range(1, 1 + ns)),
        density=STANDARD_DENSITY_FLY,
    )
    for loss_amount, letter, ns in [
        ("0.01loss", "u3", 3),
        ("0.1loss", "j3", 3),
        ("loss", "a1", 3),
    ]
    for btl in (5, 17, 65)
]

fly_fcb_ois = EndToEndModelsForTesting(
    name=f"AM [fly] + BTL + OIS",
    path_prefix=f"model/msp-310aa1",
    seeds=(1, 2, 3, 4, 5),
    density=STANDARD_DENSITY_FLY,
)

bottlenecks_with_loss_fly_selected = [
    EndToEndModelsForTesting(
        name=f"AM [fly] + BTL",
        path_prefix=f"model/msp-310a2",
        seeds=(1, 2, 3, 4, 5),
        density=STANDARD_DENSITY_FLY,
    ),
    fly_fcb_ois,
]

fly_npsc_fcb = EndToEndModelsForTesting(
    name=f"AM [fly] + BTL + NPSC",
    path_prefix=f"model/msp-310b2",
    seeds=(1, 2, 3, 4, 5),
    density=STANDARD_DENSITY_FLY,
)

fly_fcb_ois_btl1 = EndToEndModelsForTesting(
    name=f"AM [fly] + BTL + B(1)PSC + OIS",
    path_prefix=f"model/msp-310g.1a1",
    seeds=(1, 2, 3, 4, 5),
    density=STANDARD_DENSITY_FLY,
)

fly_fcb_ois_sparseprop = EndToEndModelsForTesting(
    name=f"AM [fly] + BTL + PSPSC + OIS",
    path_prefix=f"model/msp-310fa2",
    seeds=(1, 2, 3, 4, 5),
    density=STANDARD_DENSITY_FLY,
)

fly_fcb_ois_sparseprop_mul = EndToEndModelsForTesting(
    name=f"AM [fly] + BTL + PSPSC-mul + OIS",
    path_prefix=f"model/msp-310fb2",
    seeds=(1, 2, 3, 4, 5),
    density=STANDARD_DENSITY_FLY,
)

weaker_influence_models_fly = [
    fly_npsc_fcb,
    EndToEndModelsForTesting(
        name=f"AM [fly] + BTL + NPSC + OIS",
        path_prefix=f"model/msp-310c2",
        seeds=(1, 2, 3, 4, 5),
        density=STANDARD_DENSITY_FLY,
    ),
    fly_fcb_ois_sparseprop,
    fly_fcb_ois_sparseprop_mul,
    fly_fcb_ois_btl1,
    EndToEndModelsForTesting(
        name=f"AM [fly] + BTL + B(2)PSC + OIS",
        path_prefix=f"model/msp-310g.2a1",
        seeds=(1, 2, 3, 4, 5),
        density=STANDARD_DENSITY_FLY,
    ),
    EndToEndModelsForTesting(
        name=f"AM [fly] + BTL + B(4)PSC + OIS",
        path_prefix=f"model/msp-310g.4a1",
        seeds=(1, 2, 3, 4, 5),
        density=STANDARD_DENSITY_FLY,
    ),
    EndToEndModelsForTesting(
        name=f"AM [fly] + BTL + LinearConvPSC + OIS",
        path_prefix=f"model/msp-310h1",
        seeds=(1, 2, 3, 4, 5),
        density=STANDARD_DENSITY_FLY,
    ),
]
