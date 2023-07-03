import numpy as np

by_strand = {
    "+": [
        (b"N", b"\x00"),
        (b"A", b"\x01"),
        (b"C", b"\x02"),
        (b"G", b"\x03"),
        (b"T", b"\x04"),
    ],
    "-": {
        (b"N", b"\x00"),
        (b"T", b"\x01"),
        (b"G", b"\x02"),
        (b"C", b"\x03"),
        (b"A", b"\x04"),
    },
}

by_strand_exon_labels = {
    "+": {"exstart": "3'", "exend": "5'"},
    "-": {"exstart": "5'", "exend": "3'"},
}


def locate_cassette_exon(
    datafile, row, *, experimental_setting, sl=5000, cl=400, cl_max=10_000
):
    """
    Locate the cassette exon represented by row in the given splice AI datafile.

    Produces a sequence of length sl + cl, with the cassette exon in the middle.

    Parameters
    ----------
        datafile: the splice AI datafile
        row: a row from a BED file representing the cassette exon
        experimental_setting: an ExperimentalSetting object representing the kind of row it is
        sl: the sequence length to produce
        cl: the context length to produce. The cassette exon will be in the middle of the sequence
            and will not be included in the context.
        cl_max: the context length of the underlying datafile

    Returns (seq, coord)
        where seq is the sequence
        and coord is the coordinates of the casesete exons featuers
    """
    if row.geneSymbol not in datafile.names_backmap:
        # name is not found in the datafile, probably not in this train/test split
        return None
    gene_idx = datafile.names_backmap[row.geneSymbol]
    # quick checks
    if datafile.chroms[gene_idx] != row.chr:
        # clearly refering to a different gene with the same name
        return None
    if datafile.strands[gene_idx] != row.strand:
        # clearly refering to a different gene with the same name
        return None

    feature_names = sorted(experimental_setting.feature_map)

    coord = np.array(row[feature_names])
    if not (
        (datafile.starts[gene_idx] <= coord) & (coord <= datafile.ends[gene_idx])
    ).all():
        # not in the gene text as defined by spliceai
        return None

    # get parts of the sequence
    start, end = datafile.starts[gene_idx], datafile.ends[gene_idx]
    seq = datafile.datafile["SEQ"][gene_idx].upper()[cl_max // 2 + 1 : -(cl_max // 2)]
    assert len(seq) == end - start

    # relative to gene's coordinates
    if row.strand == "-":
        seq = seq[::-1]
        coord = end - np.array(coord)
    else:
        coord = np.array(coord) - start

    seq, seq_start = centered_sequence(
        seq=seq,
        middle=experimental_setting.middle(feature_names, coord),
        radius=(sl + cl) // 2,
    )

    seq = fix_strand_and_to_one_hot(seq, row.strand)

    feature_labels = [experimental_setting.feature_map[k] for k in feature_names]
    feature_labels = [
        (by_strand_exon_labels[row.strand][a], b) for a, b in feature_labels
    ]
    coord = dict(zip(feature_labels, coord - seq_start))
    for k in list(coord):
        # need to move 5' sites back one to make the coordinates consistent with spliceai style
        if k[0] == "5'":
            coord[k] -= 1

    # clip all the features
    coord = {k: v for k, v in coord.items() if cl // 2 <= v < sl + cl // 2}
    if not experimental_setting.check(coord):
        return None
    return seq, coord


def centered_sequence(*, seq, middle, radius):
    """
    Produce a centered sequence at the given point

    Parameters
    ----------
        seq: the underlying sequence
        middle: the middle of the section to cut
        radius: the radius of the section to cut

    Returns
    -------
        seq: the sequence, now clipped to size 2 * radius and centered. Padded with Ns if necessary
        seq_start: the start of the sequence in the original sequence's coordinate scheme
    """
    sl_start = max(middle - radius, 0)
    sl_end = min(middle + radius, len(seq))
    pad_left, pad_right = sl_start - (middle - radius), (middle + radius) - (sl_end)

    seq_start = middle - radius
    seq = seq[sl_start:sl_end]

    seq = b"N" * pad_left + seq + b"N" * pad_right
    assert len(seq) == radius * 2

    return seq, seq_start


def fix_strand_and_to_one_hot(seq, strand):
    """
    Fix the strand and convert to one-hot encoding.
    Only does complement, not reverse.
    """
    for a, b in by_strand[strand]:
        seq = seq.replace(a, b)
    seq = np.array(list(seq))
    seq = np.eye(4, dtype=np.float32)[seq - 1] * (seq != 0)[:, None]
    return seq
