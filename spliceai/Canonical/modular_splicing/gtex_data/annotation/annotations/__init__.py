from .constituitive import Constituitive
from .null import Null
from .alternate_sites import AlternateSites
from .skipped_exon import SkippedExon
from .exclusive_exons import ExclusiveExons
from .other import Other


annotation_types = [
    Constituitive,
    Null,
    AlternateSites,
    SkippedExon,
    ExclusiveExons,
    Other,
]
