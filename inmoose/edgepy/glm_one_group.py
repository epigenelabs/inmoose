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

# This file is a Python port of the original C++ code from the files
# 'src/glm_one_group.cpp' and 'src/R_fit_one_group.cpp' of the Bioconductor
# edgeR package (version 3.38.4).

import numpy as np

from .edgepy_cpp import glm_one_group_cython


def glm_one_group(counts, offset, disp, weights, maxit, tolerance, cur_beta):
    """
    Simplified fit for negative binomial GLM when the design matrix is one group

    This is the low-level function called by :func:`fit_one_group`.

    See Also
    --------
    fit_one_group

    Arguments
    ---------
    counts : array_like
        matrix of counts
    offset : array_like
        matrix of offsets for the log-linear models
    disp : array_like
        matrix of dispersions
    weights : array_like
        matrix of observation weights
    maxit : int
        maximum number of Newton-Raphson iterations
    tol : float
        tolerance for convergence in the Newton-Raphson iteration
    cur_beta : array_like
        initial coefficients

    Returns
    -------
    ndarray
        fitted coefficients (one per row in :code:`y`)
    ndarray
        whether the fit converged for each coefficient
    """

    out_beta = cur_beta.copy()
    out_conv = np.repeat(False, out_beta.shape)

    for j in range(out_beta.shape[0]):
        (out_beta[j], out_conv[j]) = glm_one_group_cython(
            counts.shape[1],
            counts[j, :],
            offset[j, :],
            disp[j, :],
            weights[j, :],
            maxit,
            tolerance,
            cur_beta[j],
        )
    return (out_beta, out_conv)


def fit_one_group(y, offsets, disp, weights, maxit, tol, beta, usePoisson=True):
    """
    Simplified fit for negative binomial GLM when the design matrix is one group

    In essence, this function has the same purpose as :func:`fit_levenberg`,
    specialized for the case where the design matrix is one group. In
    particular, Levenberg damping is not required here.

    This function also uses a Poisson approximation for zero dispersion genes,
    and calls :func:`glm_one_group` otherwise.

    See Also
    --------
    fit_levenberg
    glm_one_group

    Arguments
    ---------
    y : array_like
        matrix of counts
    offsets : array_like
        matrix of offsets for the log-linear models
    disp : array_like
        matrix of dispersions
    weights : array_like
        matrix of observation weights
    maxit : int
        maximum number of Newton-Raphson iterations
    tol : float
        tolerance for convergence in the Newton-Raphson iteration
    beta : array_like
        initial coefficients
    usePoisson : bool, optional
        whether to use Poisson approximation for zero-disp one-weight genes.
        Defaults to :code:`True`

    Returns
    -------
    ndarray
        fitted coefficients (one per row in :code:`y`)
    ndarray
        whether the fit converged for each coefficient
    """

    assert beta.shape == (y.shape[0],)

    out_beta = beta.copy()
    out_conv = np.repeat(False, beta.shape)

    # Checking for the Poisson special case with all-unity weights and all-zero
    # dispersions
    if usePoisson:
        disp_is_zero = (disp == 0).all(axis=1)
        weight_is_one = (weights == 1).all(axis=1)
        special_case = disp_is_zero & weight_is_one
        general_case = ~special_case
    else:
        special_case = np.repeat(False, y.shape[0])
        general_case = ~special_case

    sum_lib = np.exp(offsets[special_case]).sum(axis=1)
    sum_counts = y[special_case].sum(axis=1)
    with np.errstate(divide="ignore"):
        out_beta[special_case] = np.where(
            sum_counts == 0, -np.inf, np.log(sum_counts / sum_lib)
        )
    out_conv[special_case] = True

    # Otherwise going through NR iterations
    out_beta[general_case], out_conv[general_case] = glm_one_group(
        y[general_case],
        offsets[general_case],
        disp[general_case],
        weights[general_case],
        maxit,
        tol,
        out_beta[general_case],
    )

    return (out_beta, out_conv)
