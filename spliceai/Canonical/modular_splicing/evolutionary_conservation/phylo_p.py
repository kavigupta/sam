import os
import subprocess

import bbi

from modular_splicing.dataset.datafile_object import SpliceAIDatafile

import modular_splicing
from modular_splicing.dataset.additional_data import DatafileReferencingAdditionalData


PHYLO_P_DATA_WEB = "rsync://hgdownload.cse.ucsc.edu/goldenPath/hg19/phyloP100way/hg19.100way.phyloP100way.bw"
DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(modular_splicing.__file__))), "data"
)
PHYLO_P_DATA_LOCAL = os.path.join(DATA_PATH, "phylo_p/hg19.100way.phyloP100way.bw")


def phylo_p():
    """
    Produce phylo p bbi handle, download the data if necessary.
    """
    if not os.path.exists(PHYLO_P_DATA_LOCAL):
        subprocess.check_call(
            [
                "rsync",
                "-avz",
                "--progress",
                PHYLO_P_DATA_WEB,
                os.path.dirname(PHYLO_P_DATA_LOCAL),
            ]
        )
    return bbi.open(PHYLO_P_DATA_LOCAL)


class PhyloPAdditionalData(DatafileReferencingAdditionalData):
    """
    Compute the phyloP score for a given sequence.
    """

    def __init__(self, *args, cl_max, **kwargs):
        super().__init__(*args, **kwargs)
        self.datafiles = {
            dataset_path: SpliceAIDatafile.load(datafile_path)
            for dataset_path, datafile_path in self.datafiles.items()
        }
        self.cl_max = cl_max

    def compute_additional_input(self, original_input, path, i, j):
        aligner = self.aligners[path]

        gene_idx, loc_in_gene = aligner.get_gene_idx(i, j)
        loc_in_gene *= self._sl

        dfile = self.datafiles[path]

        chrom = dfile.chroms[gene_idx]
        start = dfile.starts[gene_idx]
        strand = dfile.strands[gene_idx]

        if strand == "-":
            loc_in_gene = dfile.ends[gene_idx] - dfile.starts[gene_idx] - loc_in_gene

        seq_start = start + loc_in_gene - self.cl_max // 2
        seq_end = seq_start + self._sl

        with phylo_p() as f:
            res = f.fetch(
                str(chrom), seq_start - self.cl_max // 2, seq_end + self.cl_max // 2
            )[:, None]
        if strand == "-":
            res = res[::-1]
        return res
