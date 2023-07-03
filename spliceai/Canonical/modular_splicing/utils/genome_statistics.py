import h5py

from permacache import permacache


@permacache("modules/additional_inputs/gene_element_counts")
def gene_element_counts(datafile_path, index, cl_max):
    """
    Get the counts of each nucleotide in a gene.

    Parameters
    ----------
    datafile_path : str
        Path to the datafile.
    index : int
        Index of the gene in the datafile.
    cl_max : int
        Amount of padding to add to the gene.

    Returns
    -------
    dict
        Counts of each nucleotide, including N.
    """
    with h5py.File(datafile_path, "r") as f:
        seq = f["SEQ"][index]
    seq = seq[cl_max // 2 : len(seq) - cl_max // 2]
    seq = seq.upper()
    gene_element_counts = {x: seq.count(x.encode("utf-8")) for x in "ACGTN"}
    assert len(seq) == sum(gene_element_counts.values())
    return gene_element_counts


def gene_at_richness(datafile_path, index, cl_max):
    """
    Get the AT richness of a gene. This is defined as the ratio of AT
    nucleotides to all ATGC nucleotides. (not including N)

    Parameters
    ----------
    datafile_path : str
        Path to the datafile.
    index : int
        Index of the gene in the datafile.
    cl_max : int
        Amount of padding to add to the gene.

    Returns
    -------
    float
        AT richness of the gene.
    """
    counts = gene_element_counts(datafile_path, index, cl_max)
    return (counts["A"] + counts["T"]) / sum(counts[x] for x in "ACGT")
