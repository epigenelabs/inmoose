import numpy as np

from edgepy_cpp import cxx_get_levenberg_start, cxx_fit_levenberg
from .makeCompressedMatrix import _compressDispersions, _compressOffsets, _compressWeights
from .utils import _isAllZero

def mglmLevenberg(y, design, dispersion=0, offset=0, weights=None, coef_start=None, start_method="null", maxit=200, tol=1e-6):
    """
    Fit genewise negative binomial GLMs with log-link using Levenberg damping to ensure convergence
    """
    # Check arguments
    y = np.asarray(y, order='F')
    (ngenes, nlibs) = y.shape
    if nlibs == 0 or ngenes == 0:
        raise ValueError("no data")

    # Check for negative, NA or non-finite values in the count matrix
    _isAllZero(y)

    # Check the design matrix
    design = np.asarray(design, order='F', dtype='double')
    # TODO check that all entries in design matrix are finite

    # Check dispersions, offsets, and weights
    offset = _compressOffsets(y, offset=offset)
    dispersion = _compressDispersions(y, dispersion)
    weights = _compressWeights(y, weights)

    # Initialize values for the coefficients at reasonable best guess with linear models
    if coef_start is None:
        if start_method not in ["null", "y"]:
            raise ValueError(f"invalid start_method {start_method}")
        beta = cxx_get_levenberg_start(y, offset, dispersion, weights, design, start_method=="null")
    else:
        beta = np.asarray(coef_start, order='F', dtype='double')

    assert beta.shape == (y.shape[0], design.shape[1])
    # Check the arguments and call the C++ method
    output = cxx_fit_levenberg(y, offset, dispersion, weights, design, beta, tol, maxit)

    # Name the output and return it
    # TODO
    return output
