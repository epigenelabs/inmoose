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

# This file is based on the file 'R/dispCoxReid.R' of the Bioconductor edgeR package (version 3.38.4).


import numpy as np
from scipy.optimize import minimize_scalar

from .adjustedProfileLik import adjustedProfileLik
from .aveLogCPM import aveLogCPM
from .makeCompressedMatrix import _compressOffsets, _compressWeights
from .systematicSubset import systematicSubset


def dispCoxReid(
    y,
    design=None,
    offset=None,
    weights=None,
    AveLogCPM=None,
    interval=(0, 4),
    tol=1e-5,
    min_row_sum=5,
    subset=10000,
):
    """
    Estimate a common dispersion parameter across multiple negative binomial
    GLMs, by maximizing the Cox-Reid adjusted profile likelihood.

    This is a low-level function called by :func:`estimateGLMCommonDisp`.

    Estimation is done by maximizing the Cox-Reid adjusted profile likelihood
    (Cox and Reid, 1987 [1]_), through :func:`scipy.optimize.minimize_scalar`.

    Robinson and Smyth (2008) [2]_ and McCarthy et al. (2012) [3]_ showed that
    the Pearson (pseudo-likelihood) estimator typically under-estimates the true
    dispersion. It can be seriously biased when the number of libraries is small.
    On the other hand, the deviance (quasi-likelihood) estimator typically
    over-estimates the true dispersion when the number of libraries is small.
    Robinson and Smyth (2008) [2]_ and McCarthy et al. (2012) [3]_ showed the
    Cox-Reid estimator to be the least biased of the three options.

    Arguments
    ---------
    y : matrix
        matrix of counts. A GLM is fitted to each row
    design : matrix, optional
        design matrix, as in :func:`glmFit`
    offset : array_like, optional
        vector or matrix of offsets for the log-linear models, as in
        :func:`glmFit`. Defaults to :code:`log(colSums(y))`.
    weights : matrix, optional
        observation weights
    AveLogCPM : array_like, optional
        vector giving average log2 counts per million
    interval : tuple, optional
        pair giving minimum and maximum allowed values for the dispersion,
        passed to :func:`scipy.optimize.minimize_scalar`
    tol : float, optional
        the desired accuracy, see :func:`scipy.optimize.minimize_scalar`
    min_row_sum : int, optional
        only rows with at least this number of counts are used
    subset : int, optional
        number of rows to use in the calculation. Rows used are chosen evenly
        space by :code:`AveLogCPM`.

    Returns
    -------
    float
        the estimated common dispersion

    References
    ----------
    .. [1] D. R. Cox, N. Reid. 1987. Parameter orthogonality and approximate
       conditional inference. Journal of the Royal Statistical Society Series B
       49, 1-39.
    .. [2] M. D. Robinson, G. K. Smyth. 2008. Small-sample estimation of
       negative binomial dispersion, with applications to SAGE data.
       Biostatistics 9, 321-332. :doi:`10.1093/biostatistics/kxm030`
    .. [3] D. J. McCarthy, Y. Chen, G. K. Smyth. 2012. Differential expression
       analysis of multifactor RNA-Seq experiments with respect to biological
       variation. Nucleic Acids Research 40, 4288-4297. :doi:`10.1093/nar/gks042`
    """
    # Check y
    y = np.asarray(y)

    # Check design
    if design is None:
        design = np.ones((y.shape[1], 1))
    else:
        design = np.asarray(design)

    # Check offset
    if offset is None:
        offset = np.log(y.sum(axis=0))
    if offset.ndim == 1:
        offset = np.broadcast_to(offset, y.shape)
    assert offset.shape == y.shape

    if interval[0] < 0:
        raise ValueError("please give a non-negative interval for the dispersion")

    if AveLogCPM is not None:
        AveLogCPM = np.asarray(AveLogCPM)

    # Apply min row count
    small_row_sum = y.sum(axis=1) < min_row_sum
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
    if subset is not None and subset <= y.shape[0] / 2:
        if AveLogCPM is None:
            AveLogCPM = aveLogCPM(y, offset=offset, weights=weights)
        i = systematicSubset(subset, AveLogCPM)
        y = y[i, :]
        offset = offset[i, :]
        if weights is not None:
            weights = weights[i, :]

    # Function for optimizing
    def sumAPL(par, y, design, offset, weights):
        return -sum(adjustedProfileLik(par**4, y, design, offset, weights=weights))

    # anticipate the calls to _compress* in adjustedProfileLik
    y = np.asarray(y)
    offset = _compressOffsets(y, offset=offset)
    weights = _compressWeights(y, weights)
    out = minimize_scalar(
        sumAPL,
        args=(y, design, offset, weights),
        bounds=(interval[0] ** 0.25, interval[1] ** 0.25),
        options={"xatol": tol},
    )
    return out.x**4
