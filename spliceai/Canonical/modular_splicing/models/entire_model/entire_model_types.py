def entire_model_types():
    from modular_splicing.models.lssi import SplicePointIdentifier
    from .preprocessed_spliceai import (
        PreprocessedSpliceAI,
    )
    from .spliceai_as_downstream import SpliceaiAsDownstream
    from .modified import (
        DroppedMotifFromModel,
        DiscretizeMotifModel,
    )
    from .modular_predictor import (
        ModularSplicePredictor,
    )
    from shelved.branch_sites.splicepoint_based_branch_site_predictor import (
        SplicepointBasedBranchSitePredictor,
    )
    from shelved.donorlike_motifs.models import WithAndWithoutAdjustedDonor
    from .reconstruct_sequence import ReconstructSequenceModel
    from .separate_downstreams import (
        SeparateDownstreams,
    )

    from modular_splicing.gtex_data.models.multi_tissue_probabilities_predictor import (
        multi_tissue_probabilities_predictor_env_types,
    )

    return dict(
        SplicePointIdentifier=SplicePointIdentifier,
        PreprocessedSpliceAI=PreprocessedSpliceAI,
        SpliceaiAsDownstream=SpliceaiAsDownstream,
        DroppedMotifFromModel=DroppedMotifFromModel,
        DiscretizeMotifModel=DiscretizeMotifModel,
        ModularSplicePredictor=ModularSplicePredictor,
        SplicepointBasedBranchSitePredictor=SplicepointBasedBranchSitePredictor,
        ReconstructSequenceModel=ReconstructSequenceModel,
        SeparateDownstreams=SeparateDownstreams,
        **multi_tissue_probabilities_predictor_env_types(),
        WithAndWithoutAdjustedDonor=WithAndWithoutAdjustedDonor,
    )
