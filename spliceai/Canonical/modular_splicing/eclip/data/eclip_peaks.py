import numpy as np


def eclips_from_onehot(all_eclip_motifs):
    """
    Produce a list of eclip peaks from a one-hot encoding of eclip peaks.

    Parameters
    ----------
    all_eclip_motifs: a matrix of shape (N, L, M, R, 2), where M is the number of motifs,
        R is the number of replicates, and 2 is the one-hot of start vs end

    Returns
    -------
    a list of eclip dictionary objects, one per peak.
    """
    clips = []
    for motif_idx in range(all_eclip_motifs.shape[2]):
        eclips_for_motif = all_eclip_motifs[:, :, motif_idx]
        for batch_idx in range(eclips_for_motif.shape[0]):
            for replicate_idx in range(all_eclip_motifs.shape[3]):
                clip_endpoints = np.array(
                    np.where(eclips_for_motif[batch_idx, :, replicate_idx])
                ).T
                assert clip_endpoints.shape[1] == 2
                for first_endpoint, second_endpoint in zip(
                    clip_endpoints[:-1], clip_endpoints[1:]
                ):
                    if first_endpoint[1] != 0 or second_endpoint[1] != 1:
                        continue
                    clips.append(
                        dict(
                            batch_idx=batch_idx,
                            motif_idx=motif_idx,
                            replicate_idx=replicate_idx,
                            start=first_endpoint[0],
                            end=second_endpoint[0],
                        )
                    )
    return clips
