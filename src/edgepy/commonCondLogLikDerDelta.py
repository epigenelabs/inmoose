from .condLogLikDerDelta import condLogLikDerDelta

def commonCondLogLikDerDelta(delta, y, der=0):
    """
    Calculate the common conditional log-likelihood (i.e. summed over all tags)
    This function is necessary so that minimization can be applied in `estimateCommonDisp`
    """
    l0 = 0
    for x in y:
        l0 += condLogLikDerDelta(delta, y=x, der=der).sum()
    return l0
