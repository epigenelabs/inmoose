# -----------------------------------------------------------------------------
# Copyright (C) 2024 Maximilien Colange

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

import copy
import logging
import re

import numpy as np
import pandas as pd
import patsy
import scipy

from ..utils import Factor, cov2cor


def makeContrasts(contrasts, levels):
    """
    Construct the contrast matrix corresponding to specified contrasts of a set of parameters.

    This function expresses contrasts between a set of parameters as a numeric
    matrix. The parameters are usually the coefficients from a linear model
    fit, so the matrix specifies which comparisons between the coefficients are
    to be extracted from the fit. The output from this function is usually used
    as input to :func:`contrasts_fit`.

    Arguments
    ---------
    contrasts : list
        strings parseable as expressions that describe the contrasts
        if :code:`contrasts` is a string, it will be interpreted as a list of
        one element
    levels : list or Factor or patsy.DesignMatrix
        list of strings or factor the names of the parameters of which
        contrasts are desired, or a design matrix or other object with the
        parameter names as column names.

    Returns
    -------
    pd.DataFrame
        matrix with columns corresponding to contrasts
    """
    if isinstance(levels, patsy.DesignMatrix):
        levels = levels.design_info.column_names
    levels = Factor(levels)

    def indicator(i, n):
        res = np.zeros(n)
        res[i] = 1
        return res

    def rename(n):
        return re.sub("([a-zA-Z0-9_]+)\[((T.)?[a-zA-Z0-9_]+)\]", "\\1_\\2", n)

    gl = {
        rename(L): indicator(i, levels.nlevels())
        for i, L in enumerate(levels.categories)
    }
    if not isinstance(contrasts, list):
        contrasts = [contrasts]
    return pd.DataFrame(
        {c: eval(rename(c), gl) for c in contrasts}, index=levels.categories
    )


def contrasts_fit(fit, contrasts=None, coefficients=None):
    """
    Compute contrasts from linear model fit

    Given a linear model fit to microarray data, compute estimated coefficients
    and standard errors for a given set of contrasts.

    This function accepts input from any of the functions :func:`lmFit`,
    :func:`lm_series`, :func:`mrlm`, :func:`gls_series` or :func:`lmscFit`. The
    function reorientates the fitted model object from the coefficients of the
    original design matrix to any set of contrasts of the original
    coefficients. The coefficients, unscaled standard deviations and
    correlation matrix are re-calculated in terms of the contrasts.

    The idea of this function is to fit a full-rank model using :func:`lmFit`
    or equivalent, then use :func:`contrasts_fit` to obtain coefficients and
    standard errors for any number of contrasts of the coefficients of the
    original model.  Unlike the design matrix input to :func:`lmFit`, which
    normally has one column for each treatment in the experiment, the matrix
    :code:`contrasts` may have any number of columns and these are not required
    to be linearly independent. Methods of assessing differential expression,
    such as :func:`eBayes` or :func:`classifyTestsF`, can then be applied to
    fitted model object.

    The :code:`coefficients` argument provides a simpler way to specify the
    :code:`contrasts` matrix when the desired contrasts are just a subset of
    the original coefficients.

    Note
    ----
    For efficiency reasons, this function does not re-factorize the design
    matrix for each probe. A consequence is that, if the design matrix is
    non-orthogonal and original fit included precision weights or missing
    values, then the unscaled standard deviations produced by this function are
    approximate rather than exact. The approximation is usually acceptable. If
    not, then the issue can be avoided by redefining the design matrix to fit
    the contrasts directly.

    Even with precision weights, the results from :func:`contrasts_fit` are
    always exact if the coefficients being compared are statistically
    independent. This will be true, for example, if the original fit was a
    oneway model without blocking and the group-means (no-intercept)
    parameterization was used for the design matrix.

    Arguments
    ---------
    fit : MArrayLM
        object produced by :func:`lm_series` or equivalent. Must contain
        components :code:`coefficients` and :code:`stdev_unscaled`.
    contrasts : array_like
        numeric matrix with rows corresponding to coefficients in :code:`fit`
        and columns containing contrasts. May be a vector if there is only one
        contrast. :code:`NA` are not allowed.
    coefficients : array_like
        array indicating which coefficients are to be kept in the revised fit
        object. An alternative way to specify the :code:`contrasts`.

    Returns
    -------
    MArrayLM
        an object of the same class as :code:`fit`, with components:

        - :code:`coefficients`: matrix containing the estimated coefficients
          for each contrast for each probe
        - :code:`stdev_unscaled`: matrix conformal with :code:`coefficients`
          containing the unscaled standard deviations for the coefficient
          estimators
        - :code:`cov_coefficients`: matrix giving the unscaled covariance
          matrix of the estimable coefficients

        Most other attributes of :code:`fit` are pass through unchanged, but
        :code:`t`, :code:`p_value`, :code:`lods`, :code:`F` and
        :code:`F_p_value` will all be removed.
    """
    fit = copy.deepcopy(fit)
    if (contrasts is None) == (coefficients is None):
        raise ValueError("must specify exactly one of contrasts or coefficients")

    if coefficients is not None:
        return fit[:, coefficients]

    if fit.coefficients is None:
        raise ValueError("fit must contain coefficients component")
    if fit.stdev_unscaled is None:
        raise ValueError("fit must contain stdev_unscaled component")

    # Remove test statistics in case eBayes() has previously been run on the fit object
    try:
        del fit.t
    except AttributeError:
        pass
    try:
        del fit.p_value
    except AttributeError:
        pass
    try:
        del fit.lods
    except AttributeError:
        pass
    try:
        del fit.F
    except AttributeError:
        pass
    try:
        del fit.F_p_value
    except AttributeError:
        pass

    # Number of coefficients in fit
    ncoef = fit.coefficients.shape[1]

    # Check contrasts
    if np.any(np.isnan(contrasts)):
        raise ValueError("contrasts must be a numeric matrix")
    if contrasts.ndim == 1:
        contrasts = contrasts[:, None]
    if contrasts.shape[0] != ncoef:
        raise ValueError(
            "Number of rows of contrast matrix must match number of coefficients in fit"
        )

    fit.contrasts = contrasts

    # Special case of contrast matrix with 0 columns
    if contrasts.shape[1] == 0:
        return fit[:, 0]

    # Correlation matrix of estimable coefficients
    # Test whether design was orthogonal
    if fit.cov_coefficients is None:
        logging.warnings.warn(
            "cov_coefficients not found in fit -- assuming coefficients are orthogonal"
        )
        var_coef = np.mean(fit.stdev_unscaled**2, axis=0)
        fit.cov_coefficients = np.diag(var_coef)
        cormatrix = np.eye(ncoef)
        orthog = True
    else:
        cormatrix = cov2cor(fit.cov_coefficients)
        if len(cormatrix) < 2:
            orthog = True
        else:
            orthog = np.sum(np.abs(np.tril(cormatrix, -1))) < 1e-12

    # if design matrix was singular, reduce to estimable coefficients
    r = cormatrix.shape[0]
    if r < ncoef:
        raise NotImplementedError()

    # Remove coefficients that do not appear in any contrast
    # (Not necessary but can make function faster)
    # TODO

    # Replace NA coefficients with large but finite standard deviations
    # to allow zero contrast entries to clobber NA coefficients
    i = np.isnan(fit.coefficients)
    NACoef = np.any(i)
    if NACoef:
        fit.coefficients[i] = 0
        fit.stdev_unscaled[i] = 1e30

    # New coefficients
    fit.coefficients = fit.coefficients @ contrasts

    # Test whether design was orthogonal
    if len(cormatrix) < 2:
        orthog = True
    else:
        orthog = np.all(np.abs(np.tril(cormatrix, -1)) < 1e-14)

    # New correlation matrix
    R = scipy.linalg.cholesky(fit.cov_coefficients)
    tmp = R @ contrasts
    fit.cov_coefficients = tmp.T @ tmp

    # New standard deviations
    if orthog:
        fit.stdev_unscaled = np.sqrt(fit.stdev_unscaled**2 @ contrasts**2)
    else:
        R = scipy.linalg.cholesky(cormatrix)
        if isinstance(cormatrix, pd.DataFrame):
            R = pd.DataFrame(R, index=cormatrix.index, columns=cormatrix.columns)
        ngenes = fit.stdev_unscaled.shape[0]
        ncont = contrasts.shape[1]
        U = np.ones((ngenes, ncont))
        U = pd.DataFrame(U, index=fit.stdev_unscaled.index, columns=contrasts.columns)
        o = np.ones((1, ncoef))
        for i in range(ngenes):
            RUC = R @ (contrasts.T * fit.stdev_unscaled.iloc[i, :]).T
            U.iloc[i, :] = np.sqrt(o @ RUC**2)
        fit.stdev_unscaled = U

    # replace NAs if necessary
    if NACoef:
        i = fit.stdev_unscaled > 1e20
        fit.coefficients[i] = np.nan
        fit.stdev_unscaled[i] = np.nan

    return fit
