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

# This file is based on the file 'R/mglmLevenberg.R' of the Bioconductor edgeR package (version 3.38.4).


import numpy as np

from .glm_levenberg import fit_levenberg
from .initialize_levenberg import get_levenberg_start
from .makeCompressedMatrix import (
    _compressDispersions,
    _compressOffsets,
    _compressWeights,
)
from .utils import _isAllZero


def mglmLevenberg(
    y,
    design,
    dispersion=0,
    offset=0,
    weights=None,
    coef_start=None,
    start_method="null",
    maxit=200,
    tol=1e-6,
):
    """
    Fit genewise negative binomial GLMs with log-link using Levenberg damping to
    ensure convergence.

    This is a low-level work-horse used by higher-level functions, especially
    :func:`glmFit`. Most users will not need to call this function directly.

    This function fits a negative binomial GLM to each row of :code:`y`. The
    row-wise GLMs all have the same design matrix but possibly different
    dispersions, offsets and weights. It is low-level in that it
    operates on atomic objects (matrices and vectors).

    This function fits an arbitrary log-linear model to each response vector.
    It implements a Levenberg-Marquardt modification of the GLM scoring
    algorithm to prevent divergence. It treats the dispersion parameter of the
    negative binomial distribution as a known input.

    Arguments
    ---------
    y : array_like
        matrix of negative binomial counts. Rows for genes and columns for
        libraries.
    design : array_like
        design matrix of the GLM. Assumed to be full column rank
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
    start_method : str
        method used to generate starting values when :code:`coef_start = None`.
        Possible values are "null" to start from the null model of equal
        expression levels or :code:`y` to use the data as starting value for the
        mean.
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

        - matrix of estimated coefficients for the linear models
        - matrix of fitted values
        - vector of residual deviances
        - number of iterations used
        - Boolean vector indicating genes for which the maximum damping was
          exceeded before convergence was achieved
    """
    # Check arguments
    y = np.asarray(y)
    (ngenes, nlibs) = y.shape
    if nlibs == 0 or ngenes == 0:
        raise ValueError("no data")

    # Check for negative, NA or non-finite values in the count matrix
    _isAllZero(y)

    # Check the design matrix
    design = np.asarray(design, dtype="double")
    # TODO check that all entries in design matrix are finite

    # Check dispersions, offsets, and weights
    offset = _compressOffsets(y, offset=offset)
    dispersion = _compressDispersions(y, dispersion)
    weights = _compressWeights(y, weights)

    # Initialize values for the coefficients at reasonable best guess with linear models
    if coef_start is None:
        if start_method not in ["null", "y"]:
            raise ValueError(f"invalid start_method {start_method}")
        beta = get_levenberg_start(
            y, offset, dispersion, weights, design, start_method == "null"
        )
    else:
        beta = np.asarray(coef_start, dtype="double")

    assert beta.shape == (y.shape[0], design.shape[1])
    # Call the actual fit
    return fit_levenberg(y, offset, dispersion, weights, design, beta, tol, maxit)
