import numpy as np
from scipy.optimize import minimize_scalar

from .adjustedProfileLik import adjustedProfileLik
from .aveLogCPM import aveLogCPM
from .makeCompressedMatrix import _compressOffsets, _compressWeights
from .systematicSubset import systematicSubset

def dispCoxReid(y, design=None, offset=None, weights=None, AveLogCPM=None, interval=(0,4), tol=1e-5, min_row_sum=5, subset=10000):
    """
    Cox-Reid APL estimator of common dispersion
    """
    # Check y
    y = np.asarray(y, order='F')

    # Check design
    if design is None:
        design = np.ones((y.shape[1], 1), order='F')
    else:
        design = np.asarray(design, order='F')

    # Check offset
    if offset is None:
        offset = np.log(y.sum(axis=0))
    if len(offset.shape) == 1:
        offset = np.full(y.shape, offset, order='F')
    assert offset.shape == y.shape

    if interval[0] < 0:
        raise ValueError("please give a non-negative interval for the dispersion")

    if AveLogCPM is not None:
        AveLogCPM = np.asarray(AveLogCPM, order='F')

    # Apply min row count
    small_row_sum = y.sum(axis=1)<min_row_sum
    if small_row_sum.any():
        y = y[np.logical_not(small_row_sum)]
        offset = offset[np.logical_not(small_row_sum)]
        if weights is not None:
            weights = weights[np.logical_not(small_row_sum)]
        if AveLogCPM is not None:
            AveLogCPM = AveLogCPM[np.logical_not(small_row_sum)]
    if y.shape[0] < 1:
        raise ValueError("no data rows with required number of counts")

    # Subsetting
    if subset is not None and subset <= y.shape[0]/2:
        if AveLogCPM is None:
            AveLogCPM = aveLogCPM(y, offset=offset, weights=weights)
        i = systematicSubset(subset, AveLogCPM)
        y = y[i,:]
        offset = offset[i,:]
        if weights is not None:
            weights = weights[i,:]

    # Function for optimizing
    def sumAPL(par, y, design, offset, weights):
        return -sum(adjustedProfileLik(par**4, y, design, offset, weights=weights))

    # anticipate the calls to _compress* in adjustedProfileLik
    y = np.asarray(y, order='F')
    offset = _compressOffsets(y, offset=offset)
    weights = _compressWeights(y, weights)
    out = minimize_scalar(sumAPL, args=(y, design, offset, weights), bounds=(interval[0]**0.25, interval[1]**0.25), tol=tol)
    return out.x**4
