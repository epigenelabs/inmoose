import numpy as np

from .makeCompressedMatrix import _compressDispersions, _compressOffsets, _compressWeights
from .utils import _isAllZero
from edgepy_cpp import cxx_fit_one_group

def mglmOneGroup(y, dispersion=0, offset=0, weights=None, coef_start=None, maxit=50, tol=1e-10, verbose=False):
    """
    Fit single-group negative-binomial glm
    """
    # Check y
    # TODO check that y is a matrix and numeric
    _isAllZero(y)

    # Check dispersion
    dispersion = _compressDispersions(y, dispersion)

    # Check offset
    offset = _compressOffsets(y, offset=offset)

    # Check starting values
    if coef_start is None:
        coef_start = np.NaN
    coef_start = np.full((y.shape[0],), coef_start, dtype='double', order='F')

    # Check weights
    weights = _compressWeights(y, weights)

    # Fisher scoring iteration
    output = cxx_fit_one_group(y, offset, dispersion, weights, maxit, tol, coef_start)

    # Convergence achieved for all tags?
    if verbose and np.count_nonzero(output[1]) > 0:
        warn("max iterations exceeded for ", np.count_nonzero(output[1]), "tags")

    return output[0]

