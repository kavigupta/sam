def should_use_motif(x):
    return any(remap_motif(x).startswith(pre) for pre in ("HN", "SR", "TRA"))


def remap_motif(x):
    return {
        "PTBP1": "HNRNPI",
        "PTBP2": "HNRNP: PTBP2",
        "PTBP3": "HNRNP: PTBP3",
        "FUS": "HNRNPP2",
        "PCBP1": "HNRNPE1",
        "PCBP2": "HNRNPE2",
    }.get(x, x)
