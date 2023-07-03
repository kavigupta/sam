from modular_splicing.psams.motif_types import motifs_types
from modular_splicing.utils.construct import construct


def get_motif_names(motif_names_source):
    """
    Get the names of the motifs for the given motif names source.
    """
    from modular_splicing.eclip.test_motifs.names import peak_names

    psam_names = {
        typ: sorted(construct(motifs_types(), dict(type=typ)))[2:]
        for typ in [
            "rbns",
            "rbrc",
            "rbns_v2p1",
            "rbns_functional",
            "rbns_v2p1_functional",
        ]
    }
    eclip_names = peak_names(replicate_category="1")
    motif_names = dict(
        rbns=sorted(psam_names["rbns"]),
        rbns_v2p1=sorted(psam_names["rbns_v2p1"]),
        rbns_functional=sorted(psam_names["rbns_functional"]),
        rbns_v2p1_functional=sorted(psam_names["rbns_v2p1_functional"]),
        eclip_18=sorted(set(psam_names["rbns"]) & set(eclip_names)),
        eclip_30=sorted(set(psam_names["rbrc"]) & set(eclip_names)),
        rbrc_91=sorted(
            (set(psam_names["rbrc"]) & set(eclip_names)) | set(psam_names["rbns"])
        ),
    )[motif_names_source]

    return motif_names
