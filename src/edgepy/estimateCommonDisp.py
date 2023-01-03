from .commonCondLogLikDerDelta import commonCondLogLikDerDelta
from .dropEmptyLevels import dropEmptyLevels
from .equalizeLibSizes import equalizeLibSizes
from .splitIntoGroups import splitIntoGroups
from .validDGEList import validDGEList
from scipy.optimize import minimize_scalar

def estimateCommonDisp_DGEList(self, tol=1e-06, rowsum_filter=5, verbose=False):
    res = validDGEList(self)
    group = res.samples.group
    lib_size = res.samples.lib_size * res.samples.norm_factors

    # TODO check whether there is replication

    out = estimateCommonDisp(res.counts, group=group, lib_size=lib_size, rowsum_filter=rowsum_filter, verbose=verbose)
    res.common_dispersion = out
    res = res.equalizeLibSizes(dispersion=out)
    res.AveLogCPM = res.aveLogCPM(dispersion=out)
    return res

def estimateCommonDisp(y, group=None, lib_size=None, tol=1e-06, rowsum_filter=5, verbose=False):
    """
    Estimate common dispersion using exact conditional likelihood
    """
    # Check y
    (ntags, nlibs) = y.shape
    if ntags == 0:
        raise ValueError("No data rows")
    if nlibs < 2:
        raise ValueError("Need at least two libraries")

    # Check group
    if group is None:
        group = np.ones(nlibs)
    if len(group) != nlibs:
        raise ValueError("Incorrect length of group")
    group = dropEmptyLevels(group)

    # Check lib_size
    if lib_size is None:
        lib_size = y.sum(axis=0)
    elif len(lib_size) != nlibs:
        raise ValueError("Incorrect length of lib_size")

    # Filter low count genes
    sel = y.sum(axis=1) > rowsum_filter
    if sel.sum() == 0:
        raise ValueError("No genes satisfy rowsum filter")

    # Start from small dispersion
    disp = 0.01
    for i in [1,2]:
        print("y = ", y)
        out = equalizeLibSizes(y, group=group, dispersion=disp, lib_size=lib_size)
        print("out = ", out)
        y_pseudo = out['pseudo_counts'][sel,]
        y_split = splitIntoGroups(y_pseudo, group=group)
        print("y_pseudo = ", y_pseudo)
        print("y_split = ", y_split)
        delta = minimize_scalar(lambda x: -commonCondLogLikDerDelta(x, y=y_split, der=0), bounds=(1e-04, 100/101), tol=tol)
        #delta = delta.maximum
        disp = delta / (1-delta)

    if verbose:
        print("Disp =", round(disp, 5), ", BCV =", round(sqrt(disp), 4))

    return disp
