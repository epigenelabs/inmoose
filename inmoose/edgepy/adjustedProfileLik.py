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

# This file is based on the file 'R/adjustedProfileLik.R' of the Bioconductor edgeR package (version 3.38.4).


import numpy as np

from .edgepy_cpp import compute_apl
from .glmFit import glmFit
from .makeCompressedMatrix import (
    _compressDispersions,
    _compressOffsets,
    _compressWeights,
)


def adjustedProfileLik(
    dispersion, y, design, offset, weights=None, adjust=True, start=None, get_coef=False
):
    """
    Compute adjusted profile log-likelihoods for the dispersion parameters of
    genewise negative binomial GLMs.

    For each row of data, compute the adjusted profile log-likelihood for the
    dispersion parameter of the negative binomial GLM. The adjusted log
    likelihood is described by McCarthy et al. (2012) [2]_ and is based on the
    method of Cox and Reid (1987) [1]_.

    The adjusted profile likelihood is an approximation to the log-likelihood
    function, conditional on the estimated values of the coefficients in the NB
    log-linear models. The conditions likelihood approach is a technique for
    adjusting the likelihood function to allow for the fact that nuisance
    parameters have to be estimated in order to evaluate the likelihood. When
    estimating the dispersion, the nuisance parameters are the coefficients in
    the log-linear models.

    This implementation calls the LAPACK library to perform the Cholesky
    decomposition during adjustment estimation.

    The purpose of :code:`start` and :code:`get_coef` is to allow hot-starting
    for multiple calls to `adjustedProfileLik`, when only :code:`dispersion` is
    altered.  Specifically, the returned GLM coefficients from one call with
    :code:`get_coef=True` can be used as the :code:`start` values for the next
    call.

    The :code:`weights` argument is interpreted in terms of averages. Each
    value of :code:`y` is assumed to be the average of :code:`n` i.i.d NB
    counts, where :code:`n` is given by the weight. This assumption can be
    generalized to fractional weights.

    Arguments
    ---------
    dispersion : float or array
        vector of dispersions
    y : matrix
        matrix of counts
    design : matrix
        design matrix
    offset : matrix or vector or float
        matrix of same shape as :code:`y` giving offsets for the log-linear
        models.  Can also be scalar or a vector of length :code:`y.shape[1]`,
        in which case it is broadcasted to the same shape as :code:`y`.
    weights : matrix, optional
        numeric matrix giving observation weights
    adjust : bool, optional
        if `True` then Cox-Reid adjustment is made to the log-likelihood.
        if `False` then the log-likelihood is returned without adjustment.
    start : matrix, optional
        numeric matrix of starting values for the GLM coefficients, to be
        passed to :func:`glmFit`.
    get_coef : bool, optional
        specifying whether fitted GLM coefficients should be returned

    Returns
    -------
    array
        vector of adjusted profile log-likelihood values is returned containing
        one element for each row of `y`.
    matrix (only if `get_coef` is `True`)
        the matrix of fitted GLM coefficients

    References
    ----------
    .. [1] D. R. Cox, N. Reid. 1987. Parameter orthogonality and approximate
       conditional inference. Journal of the Royal Statistical Society Series B
       49, 1-39.
    .. [2] D. J. McCarthy, Y. Chen, G. K. Smyth. 2012. Differential expression
       analysis of multifactor RNA-Seq experiments with respect to biological
       variation. Nucleic Acids Research 40, 4288-4297. :doi:`10.1093/nar/gks042`
    """

    # Checking counts
    y = np.asarray(y)

    # Checking offsets
    offset = _compressOffsets(y, offset=offset)

    # Checking dispersion
    dispersion = _compressDispersions(y, dispersion)

    # Checking weights
    weights = _compressWeights(y, weights)

    # Fit tagwise linear models
    fit = glmFit(
        y,
        design=design,
        dispersion=dispersion,
        offset=offset,
        prior_count=0,
        weights=weights,
        start=start,
    )
    mu = fit.fitted_values
    assert mu.dtype == np.dtype("double")

    # Compute adjusted log-likelihood
    apl = compute_apl(y, mu, dispersion, weights, adjust, design)

    # Deciding what to return
    if get_coef:
        return (apl, fit.coefficients)
    else:
        return apl
