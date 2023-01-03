import numpy as np
from warnings import warn

from .addPriorCount import addPriorCount

def predFC_DGEList(self, design, prior_count=0.125, offset=None, dispersion=None, weights=None):
    if offset is None:
        offset = self.getOffset()
    if dispersion is None:
        dispersion = self.getDispersion()
    if dispersion is None:
        dispersion = 0
        warn("dispersion set to zero")

    return predFC(y=self.counts, design=design, prior_count=prior_count, offset=offset, dispersion=dispersion, weights=weights)

def predFC(y, design, prior_count=0.125, offset=None, dispersion=0, weights=None):
    """
    Shrink log-fold-changes towards zero by augmenting data counts
    """
    from .glmFit import glmFit
    # Add prior counts in proportion to library size
    (out_y, out_offset) = addPriorCount(y, offset=offset, prior_count=prior_count)

    # Check design
    design = np.asarray(design, order='F')

    # Return matrix of coefficients on log2 scale
    g = glmFit(out_y, design, offset=out_offset, dispersion=dispersion, prior_count=0, weights=weights)
    return g.coefficients / np.log(2)
