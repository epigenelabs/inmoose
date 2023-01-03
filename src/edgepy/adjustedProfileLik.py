import numpy as np
from .makeCompressedMatrix import _compressOffsets, _compressDispersions, _compressWeights
from .glmFit import glmFit
from edgepy_cpp import cxx_compute_apl

def adjustedProfileLik(dispersion, y, design, offset, weights=None, adjust=True, start=None, get_coef=False):
    """
    Tagwise Cox-Reid adjusted log-likelihoods for the dispersion.
    Dispersion can be a scalar or a tagwise vector.
    Computationally, dispersion can also be a matrix, but the APL is still computed tagwise.
    y is a matrix: rows are genes/tags/transcripts, columns are samples/libraries.
    offset is a matrix of the same dimension as y.
    """
    # Checking counts
    y = np.asarray(y, order='F')

    # Checking offsets
    offset = _compressOffsets(y, offset=offset)

    # Checking dispersion
    dispersion = _compressDispersions(y, dispersion)

    # Checking weights
    weights = _compressWeights(y, weights)

    # Fit tagwise linear models
    fit = glmFit(y, design=design, dispersion=dispersion, offset=offset, prior_count=0, weights=weights, start=start)
    mu = fit.fitted_values
    assert mu.dtype == np.dtype('double')
    assert mu.flags.f_contiguous

    # Compute adjusted log-likelihood
    apl = cxx_compute_apl(y, mu, dispersion, weights, adjust, design)

    # Deciding what to return
    if get_coef:
        # TODO
        raise RuntimeError("unimplemented")
    else:
        return apl
