
def hat(x, intercept=True):
    #if intercept:
    #    x = # TODO cbind(1, x)
    n = x.shape[0]
    x = x.qr()
    #rowSums((Q D(1) )^2)

def filterByExpr(y, design=None, group=None, lib_size=None, min_count=10, min_total_count=15, large_n=10, min_prop=0.7):
    """
    Filter low expressed genes given count matrix
    Compute True/False index vector indicating which rows to keep
    """

    # TODO check that y is numeric

    if lib_size is None:
        lib_size = y.sum(axis=0)

    # Minimum effect sample size for any of the coefficients
    if group is None:
        if design is None:
            warn("No group or design set. Assuming all samples belong to one group.")
            MinSampleSize = y.shape[1]
        else:
            # TODO
            raise RuntimeError("Unimplemented case: 'group' is None while 'design' is not")
            h = hat(design)
            MinSampleSize = 1 / max(h)
    else:
        # TODO
        #group = as.factor(group)
        # TODO
        n = tabulate(group)
        MinSampleSize = min(n[n > 0])
    if MinSampleSize > large_n:
        MinSampleSize = large_n + (MinSampleSize - large_n) * min_prop

    # CPM cutoff
    MedianLibSize = median(lib_size)
    CPM_Cutoff = min_count / MedianLibSize * 1e6
    CPM = cpm(y, lib_size)
    tol = 1e-14
    keep_CPM = (CPM >= CPM_Cutoff).sum(axis=1) >= (MinSampleSize - tol)

    # Total count cutoff
    keep_TotalCount = y.sum(axis=1) >= (min_total_count - tol)

    return keep_CPM & keep_TotalCount

