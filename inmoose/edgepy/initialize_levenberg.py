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

# This file is a Python port of the original C++ code from the file
# 'src/R_initialize_levenberg.cpp' of the Bioconductor edgeR package (version
# 3.38.4).

import numpy as np
from scipy.linalg import qr, solve_triangular


def get_levenberg_start(y, offset, disp, weights, design, use_null):
    """
    Compute an initial value for the parameters of the NB GLM

    Arguments
    ---------
    y : array_like
        matrix of counts
    offset : array_like
        matrix of offsets, same shape as :code:`y`
    disp : array_like
        matrix of dispersions, broadcastable to the shape of :code:`y`
    weights : array_like
        matrix of observation weights
    design : array_like
        design matrix
    use_null : bool
        whether to use the null method

    Returns
    -------
    ndarray
        vector of initial values for beta
    """

    if weights is None:
        weights = 1.0

    M, N = design.shape
    K = np.minimum(M, N)
    res = np.zeros((y.shape[0], N))

    if use_null:
        Q, R = qr(design, mode="economic")
        assert Q.shape == (M, K)
        assert R.shape == (K, N)

        # computing weighted average of the count:library size ratios
        curN = np.exp(offset)
        curweight = weights * curN / (1 + disp * curN)
        curweight = np.broadcast_to(curweight, y.shape)
        sum_exprs = np.sum(y * curweight / curN, axis=1)
        sum_weight = np.sum(curweight, axis=1)
        with np.errstate(divide="ignore"):
            current = np.log(sum_exprs / sum_weight)

        with np.errstate(invalid="ignore"):
            for tag in range(y.shape[0]):
                # performing the QR decomposition and taking the solution
                res[tag] = solve_triangular(
                    R[:, :K],
                    Q.T @ np.repeat(current[tag], y.shape[1]),
                    check_finite=False,
                )

    else:
        # finding the delta
        delta = np.min([np.max(y), 1.0 / 6])
        # computing normalized log-expression values
        current = np.log(np.maximum(delta, y)) - offset
        current *= np.sqrt(weights)

        for tag in range(y.shape[0]):
            Q, R = qr(np.sqrt(weights[tag, :]) * design, mode="economic")
            assert Q.shape == (M, K)
            assert R.shape == (K, N)

            # performing the QR decomposition and taking the solution
            res[tag] = solve_triangular(
                R[:, :K], Q.T @ current[tag, :], check_finite=False
            )

    return res
