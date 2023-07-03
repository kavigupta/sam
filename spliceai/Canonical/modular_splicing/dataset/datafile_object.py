from collections import defaultdict
import h5py

import attr
import numpy as np

import tqdm.auto as tqdm


@attr.s
class SpliceAIDatafile:
    """
    Represents a SpliceAI datafile, as created by `create_spliceai_datafile`.
    """

    datafile = attr.ib()
    names = attr.ib()
    names_backmap = attr.ib()
    chroms = attr.ib()
    strands = attr.ib()
    ends = attr.ib()
    starts = attr.ib()

    @classmethod
    def load(cls, path="datafile_train_all.h5"):
        datafile = h5py.File(path, "r")
        names = np.array([x.decode("utf-8") for x in datafile["NAME"][:]])
        assert len(set(names)) == len(names)
        names_backmap = {x: i for i, x in enumerate(names)}
        return cls(
            datafile=datafile,
            names=names,
            names_backmap=names_backmap,
            chroms=np.array([x.decode("utf-8") for x in datafile["CHROM"][:]]),
            strands=np.array([x.decode("utf-8") for x in datafile["STRAND"][:]]),
            ends=np.array([int(x) for x in datafile["TX_END"][:]]),
            starts=np.array([int(x) for x in datafile["TX_START"][:]]),
        )

    def index_for_coord(self, chrom, strand, coord):
        """
        Produces the index of the given coordinate in the datafile, or None if it is not found.

        Arguments:
            chrom: The chromosome of the coordinate.
            strand: The strand of the coordinate.
            coord: A list of coordinates to search for.

        Returns:
            The index of the coordinates in the datafile (a gene that contains all of them),
                or None if it is not found.

            Raises an error if multiple genes are found.
        """
        coord = np.array(coord)
        mask = (
            (self.chroms == chrom)
            & (self.strands == strand)
            & (self.starts <= min(coord))
            & (max(coord) <= self.ends)
        )
        if mask.sum() != 1:
            return None
        [[idx]] = np.where(mask)
        return idx

    def classify_bed_rows(self, table):
        """
        Place each row of the given table into its corresponding index,
            as determined by `self.index_for_coord`.

        Arguments:
            table: A pandas.DataFrame with columns "chrom", "start", "chromStart", and "chromEnd".

        Returns:
            A dictionary from indices to lists of rows.
        """
        by_idx = defaultdict(list)
        for _, row in tqdm.tqdm(table.iterrows(), total=table.shape[0]):
            idx = self.index_for_coord(
                row.chrom, row.strand, [row.chromStart, row.chromEnd]
            )
            by_idx[idx] += [row]
        return by_idx
