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

# This file is based on the file 'R/mglmOneGroup.R' of the Bioconductor edgeR package (version 3.38.4).


import numpy as np

from ..utils import LOGGER
from .glm_one_group import fit_one_group
from .makeCompressedMatrix import (
    _compressDispersions,
    _compressOffsets,
    _compressWeights,
)
from .utils import _isAllZero


def mglmOneGroup(
    y,
    dispersion=0,
    offset=0,
    weights=None,
    coef_start=None,
    maxit=50,
    tol=1e-10,
    verbose=False,
):
    """
    Fit single-group negative-binomial GLMs genewise.

    This is a low-level work-horse used by higher-level functions, especially
    :func:`glmFit`. Most users will not need to call this function directly.

    This function fits a negative binomial GLM to each row of :code:`y`. The
    row-wise GLMs all have the same design matrix but possibly different
    dispersions, offsets and weights. It is low-level in that it
    operates on atomic objects (matrices and vectors).

    This function fits an intercept only model to each response vector. In other
    words, it treats all the libraries as belonging to one group. It implements
    Fisher scoring with a score-statistic stopping criterion for each gene.  It
    treats the dispersion parameter of the negative binomial distribution as a
    known input.  Excellent starting values are available for the null model so
    this function seldom has any problems with convergence. It is used by other
    functions to compute the overall abundance for each gene.

    Arguments
    ---------
    y : array_like
        matrix of negative binomial counts. Rows for genes and columns for
        libraries.
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
        rows should agree with :code:`y` and a single column. This argument does
        not usually need to be set as the automatic starting values perform well.
    maxit : int
        the maximum number of iterations for the Fisher scoring algorithm. The
        iteration will be stopped when this limit is reached even if the
        convergence criterion has not been satisfied.
    tol : float
        the convergence tolerance. Convergence if judged successful when the
        step size falls below :code:`tol` in absolute size.

    Returns
    -------
    ndarray
        vector of coefficients
    """
    # Check y
    # TODO check that y is a matrix and numeric
    _isAllZero(y)

    # Check dispersion
    dispersion = _compressDispersions(y, dispersion)

    # Check offset
    offset = _compressOffsets(y, offset=offset)

    # Check starting values
    if coef_start is None:
        coef_start = np.nan
    coef_start = np.full((y.shape[0],), coef_start, dtype="double")

    # Check weights
    weights = _compressWeights(y, weights)

    # Fisher scoring iteration
    output = fit_one_group(y, offset, dispersion, weights, maxit, tol, coef_start)

    # Convergence achieved for all tags?
    if np.count_nonzero(output[1]) > 0:
        LOGGER.debug(f"max iterations exceeded for {np.count_nonzero(output[1])} tags")

    return output[0]
