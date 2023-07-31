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

# This file is based on the file 'R/nbinomDeviance.R' of the Bioconductor edgeR package (version 3.38.4).


import numpy as np

from .makeCompressedMatrix import _compressDispersions, _compressWeights
from .nbdev import compute_unit_nb_deviance, nb_deviance


def nbinomDeviance(y, mean, dispersion=0, weights=None):
    """
    Residual deviances for row-wise negative binomial GLMs.

    This function computes the total residual deviance for each row of :code:`y`,
    i.e. weighted row sums of the unit deviances.

    Care is taken to ensure accurate computation in limiting cases when the
    dispersion is near zero of :code:`mean * dispersion` is very large.

    See also
    --------
    nbinomUnitDeviance

    Arguments
    ---------
    y : array_like
        matrix containing the negative binomial counts, with rows for genes and
        columns for libraries. A vector will be treated as a matrix with one row.
    mean : array_like
        matrix of expected values, of same shape as :code:`y`. A vector will be
        treated as a matrix with one row.
    dispersion : array_like
        vector or matrix of negative binomial dispersions, as in :code:`glmFit`.
        Can be a scalar, a vector of length equal to the number of rows in
        :code:`y`, or a matrix of same shape as :code:`y`.
    weights : array_like, optional
        vector or matrix of non-negative weights, as in :code:`glmFit`. Can be
        a scalar, a vector of length equal to the number of columns in :code:`y`,
        or a matrix of same shape as :code:`y`.

    Returns
    -------
    ndarray
        vector of length equal to the number of rows in :code:`y`
    """
    out = _compute_nbdeviance(
        y=y, mean=mean, dispersion=dispersion, weights=weights, dosum=True
    )
    return out


def _compute_nbdeviance(y, mean, dispersion, weights, dosum):
    y = np.asarray(y)
    mean = np.asanyarray(mean, dtype="double")
    dispersion = np.asanyarray(dispersion, dtype="double")

    # Check y. May be matrix or vector
    if y.ndim == 2:
        if mean.ndim != 2:
            raise ValueError("y is a matrix but mean is not")
    else:
        n = y.shape[0]
        y = y.reshape((1, n))
        if mean.ndim == 2:
            raise ValueError("mean is a matrix but y is not")
        else:
            if mean.shape[0] == n:
                mean = mean.reshape((1, n))
            else:
                raise ValueError("length of mean differs from that of y")

        if dispersion.ndim == 2:
            raise ValueError("dispersion is a matrix but y is not")
        else:
            if dispersion.ndim > 0 and dispersion.shape[0] == n:
                dispersion = dispersion.reshape((1, n))
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
        return nb_deviance(y, mean, weights, dispersion)
    else:
        return compute_unit_nb_deviance(y, mean, dispersion)
