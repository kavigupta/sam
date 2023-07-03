import io
import gzip
import tempfile
import subprocess

import tqdm.auto as tqdm
import pandas as pd

from BCBio import GFF

# first row of https://www.gencodegenes.org/human/
ANNOTATION_FILE = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_40/gencode.v40.annotation.gff3.gz"

gz_path = tempfile.mktemp(suffix=".gz")
subprocess.check_call(["wget", ANNOTATION_FILE, "-O", gz_path])
with gzip.open(gz_path, "rt") as f:
    data = [x for x in tqdm.tqdm(f) if x[0] == "#" or x.split("\t")[2] == "gene"]
recs = list(GFF.parse(io.StringIO("".join(data))))
rows = []
for rec in recs:
    for feat in rec.features:
        [name] = feat.qualifiers["gene_name"]
        [typ] = feat.qualifiers["gene_type"]
        if typ != "protein_coding":
            continue
        row = dict(
            name=name,
            chr=rec.id,
            strand={-1: "-", 1: "+"}[feat.strand],
            start=feat.location.start.position,
            end=feat.location.end.position,
        )
        rows.append(row)

pd.DataFrame(rows).to_csv("../data/hg38/hg38_splice_table.csv", index=False)
