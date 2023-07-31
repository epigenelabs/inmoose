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

# This file is based on the file 'R/mglmOneWay.R' of the Bioconductor edgeR
# package (version 3.38.4).
# This file contains a Python port of the original C++ code from the file
# 'src/R_get_one_way_fitted.cpp' of the Bioconductor edgeR package (version
# 3.38.4).


import numpy as np
from scipy.linalg import solve

from ..utils import Factor, asfactor
from .makeCompressedMatrix import (
    _compressDispersions,
    _compressOffsets,
    _compressWeights,
)
from .mglmOneGroup import mglmOneGroup


def designAsFactor(design):
    """
    Convert a general design matrix into a oneway layout if that is possible.

    This function determines how many distinct row values the design matrix is
    capable of computing and returns a factor with a level for each possible
    distinct value.

    Arguments
    ---------
    design : matrix
        the design matrix. Assumed to be full column rank.

    Returns
    -------
    Factor
        factor of length equal to the number of rows in :code:`design`
    """
    design = np.asarray(design)
    z = (np.e + np.pi) / 5
    col = np.full(design.shape, np.arange(design.shape[1]))
    means = (design * z**col).mean(axis=1)
    uniq = np.unique(means)
    fact = np.zeros(means.shape, dtype="int64")
    for i in range(len(uniq)):
        fact[means == uniq[i]] = i + 1
    return Factor(fact)


def mglmOneWay(
    y,
    design=None,
    group=None,
    dispersion=0,
    offset=0,
    weights=None,
    coef_start=None,
    maxit=50,
    tol=1e-10,
):
    """
    Fit multiple negative binomial GLMs with log-link by Fisher scoring with a
    single explanatory factor in the model.

    This is a low-level work-horse used by higher-level functions, especially
    :func:`glmFit`. Most users will not need to call this function directly.

    This function fits a negative binomial GLM to each row of :code:`y`. The
    row-wise GLMs all have the same design matrix but possibly different
    dispersions, offsets and weights. It is low-level in that it
    operates on atomic objects (matrices and vectors).

    This function fits a oneway layout to each response vector. It treats the
    libraries as belonging to a number of groups and calls :func:`mglmOneGroup`
    for each group. It treats the dispersion parameter of the negative binomial
    distribution as a known input.

    Arguments
    ---------
    y : array_like
        matrix of negative binomial counts. Rows for genes and columns for
        libraries.
    design : array_like, optional
        design matrix of the GLM. Assumed to be full column rank. Defaults to
        :code:`~ 0 + group` if :code:`group` is specified, otherwise to
        :code:`~ 1`.
    group : :obj:`Factor`
        group memberships for oneway layout. If both :code:`design` and
        :code:`group` are specified, then they must agree in terms of
        :func:`designAsFactor`. If :code:`design = None`, then a group-means
        design matrix is implied.
    dispersion : float or array_like
        scalar or vector giving the dispersion parameter for each GLM. Can be a
        scalar giving one value for all genes, or a vector of length equal to
        the number of genes giving genewise dispersions.
    offset : array_like
        vector or matrix giving the offset that is to be included in the log
        linear model predictor. Can be a scalar, a vector of length equal to the
        number of libraries, or a matrix of the same shape as :code:`y`.
    weights : matrix, optional
        vector or matrix of non-negative quantitative weights. Can be a vector
        of length equal to the number of libraries, or a matrix of the same
        shape as :code:`y`.
    coef_start : array_like, optional
        matrix of starting values for the linear model coefficient. Number of
        rows should agree with :code:`y` and number of columns should agree with
        :code:`design`. This argument does not usually need to be set as the
        automatic starting values perform well.
    maxit : int
        the maximum number of iterations for the Fisher scoring algorithm. The
        iteration will be stopped when this limit is reached even if the
        convergence criterion has not been satisfied.
    tol : float
        the convergence tolerance.

    Returns
    -------
    tuple
        tuple with the following components:

        - matrix of estimated coefficients for the linear models. Rows
          corrspond to row of :code:`y` and columns to columns of :code:`design`
        - matrix of fitted values. Same shape as :code:`y`.
    """
    y = np.asarray(y)
    (ngenes, nlibs) = y.shape

    offset = _compressOffsets(y, offset=offset)
    dispersion = _compressDispersions(y, dispersion)
    weights = _compressWeights(y, weights)

    # If necessary, the group factor is computed from the design matrix.
    # However, if group is supplied, we can avoid creating a design matrix altogether.
    if group is None:
        if design is None:
            group = Factor(np.ones((nlibs,), dtype=np.int64))
        else:
            design = np.asarray(design)
            group = designAsFactor(design)
    else:
        group = asfactor(group)

    # Convert factor to integer levels for efficiency
    levg = group.categories
    ngroups = len(levg)
    i = group.__array__()

    if design is not None:
        if design.shape[1] != ngroups:
            raise ValueError("design matrix is not equivalent to a oneway layout")

    # Reduce to representative design matrix, based on the column in which each group appears first
    firstjofgroup = [(i == x).nonzero()[0][0] for x in levg]
    if design is not None:
        designunique = design[firstjofgroup, :]
    else:
        designunique = None

    # Is it just a group indicator matrix?
    if (
        np.sum(designunique == 1) == ngroups
        and np.sum(designunique == 0) == (ngroups - 1) * ngroups
    ):
        design = None

    # If necessary, convert starting values to group fitted values
    if design is not None and coef_start is not None:
        coef_start = coef_start @ designunique.T

    # Cycle through groups
    beta = np.zeros((ngenes, ngroups), dtype="double")
    for g in range(ngroups):
        j = np.nonzero(i == levg[g])[0]
        beta[:, g] = mglmOneGroup(
            y[:, j],
            dispersion=dispersion[:, j],
            offset=offset[:, j],
            weights=weights[:, j] if weights is not None else None,
            coef_start=coef_start[:, g] if coef_start is not None else None,
            maxit=maxit,
            tol=tol,
        )

    # Reset -inf values to finite values to simplify calculations downstream
    beta = np.where(beta > -1e8, beta, -1e8)

    # Fitted values from group-wise beta's
    mu = get_one_way_fitted(beta, offset, i - 1)

    # If necessary, reformat the beta's to reflect the original design.
    if design is not None:
        beta = solve(designunique, beta.T).T

    return (beta, mu)


def get_one_way_fitted(beta, offset, groups):
    """
    Get fitted values from a one-way layout

    Arguments
    ---------
    beta : array_like
        matrix of coefficients
    offset : array_like
        matrix of offsets
    groups : array_like
        vector of groups: one element per library, giving the index of the
        corresponding column of :code:`beta`

    Returns
    -------
    ndarray
        reconstructed fitted values. Same shape as :code:`offset`
    """

    num_groups = beta.shape[1]

    if np.min(groups) < 0:
        raise ValueError("smallest value of group vector should be non-negative")
    if np.max(groups) >= num_groups:
        raise ValueError(
            "largest value of group vector should be less than the number of groups"
        )

    # output[i,j] = exp(offset[i,j] + beta[i,groups[j]])
    return np.exp(offset + beta[:, groups])
