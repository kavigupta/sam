def find_pattern(sites, usable, skippable, pattern):
    """
    Find the sites that match the pattern, where usable and skippable are
    boolean arrays indicating which sites are usable and skippable.

    The pattern is a string of A and D, where A means "acceptor" and D means
    "donor". The pattern must consist of usable sites, but must match the
    sites in consecutive order, but may skip sites that are skippable.

    Parameters
    ----------
    sites : list of str
        List of sites, in order, where ecach site is "A" or "D".
    usable : list of bool
        List of booleans indicating which sites are usable.
    skippable : list of bool
        List of booleans indicating which sites are skippable.

    Returns
    -------
    list of tuple of int
        List of tuples of indices into the sites that match the pattern.
    """
    assert len(sites) == len(usable) == len(skippable)
    assert all(x in "AD" for x in pattern)
    assert all(x in "AD" for x in sites)
    # all non-usable sites are skippable
    assert all(u or s for u, s in zip(usable, skippable))

    def find(pattern, start, in_skip):
        if len(pattern) == 0:
            yield ()
            return
        if start == len(sites):
            return
        # with start
        if usable[start]:
            pattern_first, *pattern_rest = pattern
            if sites[start] == pattern_first:
                for rest in find(pattern_rest, start + 1, True):
                    yield (start,) + rest
        # without start
        if not in_skip or skippable[start]:
            yield from find(pattern, start + 1, in_skip)

    return list(find(pattern, 0, False))
