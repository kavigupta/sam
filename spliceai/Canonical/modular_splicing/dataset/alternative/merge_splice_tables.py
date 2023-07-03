import pandas as pd


def splice_tables_on_common_genes(data_txt_file_paths, common_genes_key):
    """
    Produces splice tables for all the given data_txt_file_paths, but only for the
        genes that appear in the table indexed by common_genes_key.

    Parameters
    ----------
    data_txt_file_paths : dict[key -> path]
        A dictionary mapping keys to paths to splice table files.
    common_genes_key : str
        The key of the data_txt_file_paths dictionary that contains the
        genes we want to keep.

    Returns (main_data, splice_tables, extra_genes)
    -------
    main_data : pd.DataFrame
        The original splice table for `common_genes_key`, pre-sorting
    splice_tables: dict[key -> pd.DataFrame]
        A dictionary mapping keys to splice tables, but only for the genes
        that appear in `main_data`. In sorted order, with genes only appearing
        once.
    extra_genes : dict[key -> set]
        A dictionary mapping keys to the genes that appear in the given splice
        table, but not in `main_data`.

    """
    data_txt_file_contents = {
        k: pd.read_csv(
            data_txt_file_paths[k],
            sep="\t",
            names=[
                "Name",
                "chunk",
                "chrom",
                "strand",
                "start",
                "end",
                "sstarts",
                "sends",
            ],
        )
        for k in data_txt_file_paths
    }

    canonical_data = data_txt_file_contents[common_genes_key].copy()

    data_txt_file_contents = {
        k: group_by_gene(data_txt_file_contents[k]) for k in data_txt_file_paths
    }
    # check that the genes in different files are identical
    _ = group_by_gene(pd.concat(data_txt_file_contents.values()))
    genes_each = {
        k: set(data_txt_file_contents[k].Name) for k in data_txt_file_contents
    }
    extra_genes = {k: genes_each[k] - genes_each[common_genes_key] for k in genes_each}
    data_txt_files_contents = {
        k: data_txt_file_contents[k][
            data_txt_file_contents[k].Name.apply(
                lambda x: x in genes_each[common_genes_key]
            )
        ]
        for k in data_txt_file_contents
    }
    return canonical_data, data_txt_files_contents, extra_genes


def group_by_gene(frame):
    """
    Group splice table together by the gene name, concatenating the lists of splice sites.
    """
    grouped = frame.groupby("Name")
    return grouped.agg(
        {
            k: sum if k in {"sstarts", "sends"} else only_one
            for k in frame
            if k != "Name"
        }
    ).reset_index()


def only_one(x):
    """
    Checks that all the elements in x are the same, and returns that element.
    """
    x = set(x)
    assert len(x) == 1
    return list(x)[0]
