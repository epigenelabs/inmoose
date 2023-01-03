import numpy as np
from .makeCompressedMatrix import _compressDispersions, _compressWeights
from edgepy_cpp import cxx_compute_nbdev_sum, cxx_compute_nbdev_nosum

def nbinomDeviance(y, mean, dispersion=0, weights=None):
    """
    Residual deviances for row-wise negative binomial GLMs.
    y is a matrix and a deviance is computed for each row.
    A vector y is taken to be a matrix with one row; in this case mean and dispersion must also be vectors
    """
    out = _compute_nbdeviance(y=y, mean=mean, dispersion=dispersion, weights=weights, dosum=True)
    return out

def _compute_nbdeviance(y, mean, dispersion, weights, dosum):
    y = np.asarray(y, order='F')
    mean = np.asanyarray(mean, order='F', dtype='double')
    dispersion = np.asanyarray(dispersion, order='F', dtype='double')

    # Check y. May be matrix or vector
    if len(y.shape) == 2:
        if len(mean.shape) != 2:
            raise ValueError("y is a matrix but mean is not")
    else:
        n = y.shape[0]
        y = y.reshape((1,n))
        if len(mean.shape) == 2:
            raise ValueError("mean is a matrix but y is not")
        else:
            if mean.shape[0] == n:
                mean = mean.reshape((1,n))
            else:
                raise ValueError("length of mean differs from that of y")

        if len(dispersion.shape) == 2:
            raise ValueError("dispersion is a matrix but y is not")
        else:
            if len(dispersion.shape) > 0 and dispersion.shape[0] == n:
                dispersion = dispersion.reshape((1,n))
            else:
                raise ValueError("length of dispersion differs from that of y")

    # Check mean
    if y.shape != mean.shape:
        raise ValueError("mean should have the same dimensions as y")
    mean = mean.astype(np.float64, copy=False)

    # Check dispersion (can be tagwise (i.e. rowwise) or observation-wise)
    dispersion = _compressDispersions(y, dispersion)

    # Check weights
    weights = _compressWeights(y, weights)

    # Compute matrix of unit deviances, or residual deviance per gene, depending on `dosum`
    if dosum:
        return cxx_compute_nbdev_sum(y, mean, dispersion, weights)
    else:
        return cxx_compute_nbdev_nosum(y, mean, dispersion, weights)

