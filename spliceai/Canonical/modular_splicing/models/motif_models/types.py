def motif_model_types():
    from .learned_motif_model import (
        LearnedMotifModel,
    )
    from .no_motif_model import NoMotifModel
    from .parallel_motif_model import ParallelMotifModels
    from .adjusted_motif_model import AdjustedMotifModel
    from .reprocessed_motif_model.reprocessed_motifs import ReprocessedMotifModel
    from shelved.donorlike_motifs.models import (
        AdjustedSplicepointModel,
        FromLoadedSplicepointModel,
    )
    from .pretrained_motif_model import PretrainedMotifModel
    from modular_splicing.models.motif_models.passthrough_motif_model import (
        PassthroughMotifsFromData,
    )

    from .neural_fixed_motif_model import NeuralFixedMotif

    from .psam_fixed_motif import PSAMMotifModel

    from .use_inputs_in import UseInputsIn
    from .compute_extra_input_then import ComputeExtraInputThen
    from .flanked_motif_model import FlankedMotifModel
    from shelved.robustly_adjusted.robustly_adjusted_motifs import (
        RobustlyAdjustedMotifs,
    )
    from shelved.donorlike_motifs.donor_separating_motif import (
        AdjustedAlternateDonorSeparatingMotifModel,
        AlternateDonorSeparatingMotifModel,
    )

    from modular_splicing.eclip.trained_on_eclip.train import PretrainedEclipMotifModel
    from shelved.motif_model_multiple_width import MotifModelMultipleWidths
    from shelved.uniqueness.models import MotifModelForRobustDownstream

    from .multiple_splicepoint_models import MultipleSplicepointModels

    return dict(
        LearnedMotifModel=LearnedMotifModel,
        PSAMMotifModel=PSAMMotifModel,
        ParallelMotifModels=ParallelMotifModels,
        AdjustedMotifModel=AdjustedMotifModel,
        ReprocessedMotifModel=ReprocessedMotifModel,
        AdjustedSplicepointModel=AdjustedSplicepointModel,
        NeuralFixedMotif=NeuralFixedMotif,
        PretrainedMotifModel=PretrainedMotifModel,
        PassthroughMotifsFromData=PassthroughMotifsFromData,
        MotifModelForRobustDownstream=MotifModelForRobustDownstream,
        NoMotifModel=NoMotifModel,
        FromLoadedSplicepointModel=FromLoadedSplicepointModel,
        AlternateDonorSeparatingMotifModel=AlternateDonorSeparatingMotifModel,
        AdjustedAlternateDonorSeparatingMotifModel=AdjustedAlternateDonorSeparatingMotifModel,
        RobustlyAdjustedMotifs=RobustlyAdjustedMotifs,
        FlankedMotifModel=FlankedMotifModel,
        ComputeExtraInputThen=ComputeExtraInputThen,
        UseInputsIn=UseInputsIn,
        PretrainedEclipMotifModel=PretrainedEclipMotifModel,
        MotifModelMultipleWidths=MotifModelMultipleWidths,
        MultipleSplicepointModels=MultipleSplicepointModels,
    )
