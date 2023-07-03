import pandas as pd

from .gtex_data_table import gene_expr_data, transcript_data


def produce_splice_table(limit=None):
    """
    Produces the hg38 splice table, filtered for genes with transcripts. Has paralog
    information added to the table.

    Parameters
    ----------
    limit : int, optional
        If not None, only the first `limit` genes in the splice table will be
        considered. For debugging purposes.

    Returns
    -------
    valid_splice_table : pd.DataFrame
        The splice table, filtered for genes with transcripts.
    name_to_ensg : dict
        A dictionary mapping gene names to ensembl gene ids.
    frac_kept : float
        The fraction of genes in the splice table that have transcripts.
    """
    splice_table = pd.read_csv("../data/hg38/hg38_splice_table.csv")
    if limit is not None:
        splice_table = splice_table.iloc[:limit].copy()
    genes = gene_expr_data()
    transcripts = transcript_data()

    name_to_ensg = dict(zip(genes.meta_1, genes.meta_0))
    ensg_containing_transcripts = set(transcripts.meta_1)
    genes_containing_transcripts = {
        g for g, e in name_to_ensg.items() if e in ensg_containing_transcripts
    }
    frac_kept = len(genes_containing_transcripts & set(splice_table.name)) / len(
        splice_table.name
    )

    valid_splice_table = splice_table[
        splice_table.name.apply(lambda x: x in genes_containing_transcripts)
    ].copy()

    paralogs = set(
        pd.read_csv("../data/hg38/ensemble_biomart_paralogs_GRCh38.p13.txt", sep="\t")[
            "Gene stable ID"
        ]
    )
    is_paralog = valid_splice_table.name.apply(
        lambda x: int(name_to_ensg[x].split(".")[0] in paralogs)
    )
    valid_splice_table.insert(1, "chunk", is_paralog)

    # get rid of chrM and chrY
    valid_splice_table = valid_splice_table[
        ~valid_splice_table.chr.isin(["chrM", "chrY"])
    ].copy()

    splice_table_names = set(valid_splice_table.name)
    name_to_ensg = {k: v for k, v in name_to_ensg.items() if k in splice_table_names}

    return valid_splice_table, name_to_ensg, frac_kept
