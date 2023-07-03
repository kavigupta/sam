from functools import lru_cache
import numpy as np

from working.synthetic_data.splicing_mechanism.motif.psam_motif import PSAMMotif
from working.synthetic_data.splicing_mechanism.basic_splicing_mechanism import (
    BasicSplicingMechanism,
    BasicEffect,
)


@lru_cache(maxsize=None)
def mechanism():
    rng = np.random.RandomState(1)

    num_motifs = 10
    num_effects = 20

    motifs = [
        PSAMMotif.sample(
            rng,
            density=0.1 / num_motifs,
            num_psams=3,
            motif_width=7,
            range=1,
        )
        for _ in range(num_motifs)
    ]

    effects = [
        x
        for _ in range(num_effects)
        for x in [
            BasicEffect(
                rng.choice(num_motifs),
                rng.randint(2),
                rng.rand() - 0.5,
                start_disp=-rng.randint(10, 200),
                end_disp=0,
                highpoint="right",
            ),
            BasicEffect(
                rng.choice(num_motifs),
                rng.randint(2),
                rng.rand() - 0.5,
                start_disp=0,
                end_disp=rng.randint(10, 200),
                highpoint="left",
            ),
        ]
    ]

    splice_mech = BasicSplicingMechanism(
        [
            PSAMMotif.sample(
                rng,
                density=0.001,
                num_psams=3,
                motif_width=7,
                activation_point=ap,
                range=1,
            )
            for ap in [6, 0]
        ],
        motifs,
        effects,
        1,
    )
    return splice_mech
