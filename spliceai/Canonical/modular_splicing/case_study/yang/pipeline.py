import functools
import io

from permacache import permacache
import pandas as pd
import tqdm.auto as tqdm

from modular_splicing.utils.download import read_gzip

EXONS_URL = "https://github.com/gxiaolab/BEAPR/raw/master/souce_data_files/f4ab.gz"
READ_COUNTS_URL = "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-019-09292-w/MediaObjects/41467_2019_9292_MOESM5_ESM.xlsx"


@permacache("modular_splicing/case_study/yang/pipeline/exons_table")
def exons_table():
    exons = read_gzip(EXONS_URL)
    exons = pd.read_csv(io.BytesIO(exons), sep="\t")
    exons[["chr", "start", "end", "strand"]] = exons["exon"].str.split(":", expand=True)
    exons["start"] = exons["start"].apply(int)
    exons["end"] = exons["end"].apply(int)
    return exons.copy()


@permacache("modular_splicing/case_study/yang/pipeline/asb_table")
def asb_table():
    asb = pd.read_excel(READ_COUNTS_URL)
    asb = asb.copy()
    asb.columns = asb.iloc[1]
    asb = asb[2:]
    return asb.copy()


@permacache("modular_splicing/case_study/yang/pipeline/attach_exons_to_asb")
def attach_exons_to_asb():
    exons = exons_table()
    asb = asb_table()

    @functools.lru_cache(None)
    def exons_for(chrom, strand):
        return exons[(exons.chr == chrom) & (exons.strand == strand)]

    exons_per_asb = []
    for _, row in tqdm.tqdm(list(asb.iterrows())):
        exons_filtered = exons_for(row.Chr, row.strand)
        exons_filtered = exons_filtered[
            (exons_filtered.RBP == row.RBP)
            & (exons_filtered.start <= row.Coordinate)
            & (row.Coordinate <= exons_filtered.end)
        ]
        exons_per_asb.append(exons_filtered)
    return exons_per_asb
