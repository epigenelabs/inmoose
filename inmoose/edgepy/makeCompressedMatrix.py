# -----------------------------------------------------------------------------
# Copyright (C) 2008-2022 Yunshun Chen, Aaron TL Lun, Davis J McCarthy, Matthew E Ritchie, Belinda Phipson, Yifang Hu, Xiaobei Zhou, Mark D Robinson, Gordon K Smyth
# Copyright (C) 2022-2023 Maximilien Colange

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------

# This file is based on the file 'R/makeCompressedMatrix.R' of the Bioconductor edgeR package (version 3.38.4).


import numpy as np


# NB: we do not fully implement edgeR class CompressedMatrix: we use fully-expanded 2D numpy arrays instead
#     still, to avoid useless calls to `_compress*` and `check_finite`, we subclass ndarray
#     note that the name `CompressedMatrix` for the subclass is only historical
class CompressedMatrix(np.ndarray):
    def __new__(cls, input_array):
        # Input array is an already formed ndarray instance
        # We thus cast to be our class type
        return np.asarray(input_array).view(cls)


def makeCompressedMatrix(x, dims, byrow=True):
    """
    Coerce a None, scalar, vector or matrix to a compressed matrix.
    """
    if x.__class__ == CompressedMatrix:
        return x

    if len(dims) != 2:
        raise ValueError("dims does not represent the shape of a matrix")
    x = np.asarray(x)
    if x.ndim > 2:
        raise ValueError("input has too many dimensions to be interpreted as a matrix")
    if x.ndim == 1 and not byrow:
        x = x[:, None]
    try:
        x = np.broadcast_to(x, dims)
    except ValueError:
        if byrow:
            raise ValueError("dims[1] should be equal to length of x")
        else:
            raise ValueError("dims[0] should be equal to length of x")

    return CompressedMatrix(x)


def check_finite(x, what, negative_allowed):
    xmin = np.amin(x)
    if np.isnan(xmin):
        raise ValueError("NaN " + what + " not allowed")
    if not negative_allowed and xmin < 0:
        raise ValueError("negative " + what + " not allowed")
    if np.isinf(np.nanmax(x)):
        raise ValueError("infinite " + what + " is not allowed")


def _compressOffsets(y, offset, lib_size=None):
    """
    Check for finite values
    If provided, offset takes precedence over lib_size
    If neither are provided, lib_size is automatically as the sum of counts in the count matrix y
    """
    if offset.__class__ == CompressedMatrix:
        return offset

    if offset is None:
        if lib_size is None:
            lib_size = y.sum(axis=0)
        offset = np.log(lib_size)

    offset = np.asarray(offset, dtype="double")
    offset = makeCompressedMatrix(offset, y.shape, byrow=True)
    check_finite(offset, "offset", negative_allowed=True)
    return offset


def _compressWeights(y, weights=None):
    """
    Check for non-negative finite values
    All weights default to 1 when not specified
    """
    if weights.__class__ == CompressedMatrix:
        return weights

    if weights is None:
        weights = 1

    weights = np.asarray(weights, dtype="double")
    weights = makeCompressedMatrix(weights, y.shape, byrow=True)
    check_finite(weights, "weights", negative_allowed=False)
    return weights


def _compressDispersions(y, dispersion):
    """
    Check for non-negative finite values
    """
    if dispersion.__class__ == CompressedMatrix:
        return dispersion

    dispersion = np.asarray(dispersion, dtype="double")
    dispersion = makeCompressedMatrix(dispersion, y.shape, byrow=False)
    check_finite(dispersion, "dispersion", negative_allowed=False)
    return dispersion


def _compressPrior(y, prior_count):
    """
    Check for non-negative finite values
    """
    if prior_count.__class__ == CompressedMatrix:
        return prior_count

    prior_count = np.asarray(prior_count, dtype="double")
    prior_count = makeCompressedMatrix(prior_count, y.shape, byrow=False)
    check_finite(prior_count, "prior counts", negative_allowed=False)
    return prior_count
