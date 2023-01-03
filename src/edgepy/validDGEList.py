def validDGEList(y):
    """
    Check for standard components of DGEList object
    Modifies y in place
    """
    if y.counts is None:
        raise RuntimeError("No count matrix")
    # TODO
    #y.counts = as.matrix(y.counts)
    nlib = y.counts.shape[1]
    if (y.samples.group.values == None).any():
        y.samples.group = gl(1, nlib)
    if (y.samples.lib_size.values == None).any():
        y.samples.lib_size = y.counts.sum(axis=0)
    if (y.samples.norm_factors.values == None).any():
        y.samples.norm_factors = np.ones(nlib)

    return y

