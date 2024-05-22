# -----------------------------------------------------------------------------
# Copyright (C) 2004-2022 Gordon Smyth, Yifang Hu, Matthew Ritchie, Jeremy Silver, James Wettenhall, Davis McCarthy, Di Wu, Wei Shi, Belinda Phipson, Aaron Lun, Natalie Thorne, Alicia Oshlack, Carolyn de Graaf, Yunshun Chen, Mette Langaas, Egil Ferkingstad, Marcus Davy, Francois Pepin, Dongseok Choi
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

# This file is based on the file 'R/lmfit.R' of the Bioconductor limma package (version 3.55.1).

import logging

import numpy as np
import pandas as pd
import patsy
import scipy

from ..utils import lm_fit, lm_wfit
from .dups import uniquegenelist, unwrapdups
from .marraylm import MArrayLM


class EAWP:
    def __init__(self, obj):
        if not isinstance(obj, pd.DataFrame):
            self.exprs = pd.DataFrame(obj)
        else:
            self.exprs = obj.copy()
        self.design = None
        self.weights = None
        self.probes = None
        self.Amean = self.exprs.mean(axis=1)


def lmFit(
    obj,
    design=None,
    ndups=None,
    spacing=None,
    block=None,
    correlation=None,
    weights=None,
    method="ls",
):
    """
    Fit linear models for each gene given a series of arrays

    This function fits multiple linear models by weighted or generalized least
    squares. It accepts data from an experiment involving a series of
    microarrays with the same set of probes. A linear model is fitted to the
    expression data of each probe. The expression data should be log-ratios for
    two-color array platforms or log-expression values for one-channel
    platforms. To fit linear models to the individual channels of two-color
    array data, see :func:`lmscFit`. The coefficients of the fitted models
    describe the differences between the RNA sources hybridized to the arrays.
    The probe-wise fitted model results are stored in a compact form suitable
    for further processing by other functions in the limma package.

    This function allows for missing values and accepts quantitative precision
    weights through the :code:`weights` argument. It also supported two
    different correlation structures. If :code:`block` is not :code:`None`,
    then different arrays are assumed to be correlated. If :code:`block` is
    :code:`None` and :code:`ndups` is greater than one then replicate spots on
    the same array are assumed to be correlated. It is not possible at this
    time to fit models with a block structure and a duplicate-spot correlation
    structure simultaneously.

    If :code:`obj` is a matrix then it should contain log-ratios or
    log-expression data with rows corresponding to probes and columns to
    arrays. A vector is treated the same as a matrix with a single column. For
    objects of other classes, a matrix of epression values is taken from the
    appropriate component or slot of the object. If :code:`obj` is of class
    :code:`MAList` or :code:`marrayNorm`, then the matrix of log-ratios
    (M-values) is extracted.

    The arguments :code:`design`, :code:`ndups`, :code:`spacing` and
    :code:`weights` will be extracted from the data :code:`obj` if available.
    On the other hand, if any of these are set to a non-:code:`None` value in
    the function call then this value will override the value found in
    :code:`obj`.

    If the argument :code:`block` is used, then it is assumed that :code:`ndups=1`.

    The :code:`correlation` argument has a default value of :code:`0.75`, but
    in normal use this default value should not be relied on and the
    correlation value should be estimated using the function
    :code:`duplicateCorrelation`. The default value is likely to be too high in
    particular if used with the :code:`block` argument.

    The actual linear model computations are done by passing the data to one of
    the lower-level functions :func:`lm_series`, :func:`gls_series` or
    :func:`mrlm`. The function :func:`mrlm` is used if :code:`method="robust"`.
    If :code:`method="ls"`, then :func:`gls_series` is used if a correlation
    structure has been specified, *i.e.* if :code:`ndups > 1` or :code:`block`
    is non-null and :code:`correlation` is different from zero. If
    :code:`method="ls"` and there is no correlation structure,
    :code:`lm_series` is used.

    An overview of linear model functions in limma is given by :ref:`linearmodels`.

    See Also
    --------
    getEAWP
        extract expression values, gene annotation and so from the data :code:`obj`.

    Arguments
    ---------
    obj : matrix-like
        a matrix-like data object containing log-ratios or log-expression
        values for a series of arrays, with rows corresponding to genes and
        columns to samples. Any type of data object that can be processed by
        :func:`getEAWP` is acceptable.
    design : patsy formula-like
        the design matrix of the microarray experiment, with rows corresponding
        to samples and columns to coefficients to be estimated.
        Defaults to :code:`obj.design` if not :code:`None`, otherwise to the
        unit vector, meaning that all samples will be treated as replicated of
        a single treatment group.
    ndups : int
        positive integer giving the number of times each distinct probe is
        printed on each array
    spacing : int
        positive integer giving the spacing between duplicate occurrences of
        the same probe, :code:`spacing=1` for consecutive rows.
    block : array-like
        vector or factor specifying a blocking variable on the arrays. Has
        length equal to the number of arrays. Must be :code:`None` if
        :code:`ndups > 2`.
    correlation
        the inter-duplicate or inter-technical replicate correlation
    weights
        non-negative precision weights. Can be a numeric matrix of individual
        weights of same size as the object expression matrix, or a numeric
        vector of gene weights with length equal to :code:`nrow` of the
        expression matrix.
    method : { "ls", "robust" }
        fitting method: :code:`"ls"` for least squares or :code:`"robust"` for
        robust regression

    Returns
    -------
    MArrayLM
        object containing the result of the fits.
        The row names of :code:`obj` are preserved in the fit object and can be
        retrieved by :code:`fit.index` where :code:`fit` is the output of
        :func:`lmFit`.
        The column names of :code:`design` are preserved as column names and
        can be retrieved by :code:`fit.columns`.
    """
    # Fit genewise linear models

    # Extract components from obj
    y = EAWP(obj)

    if y.exprs.shape[0] == 0:
        raise ValueError("expression matrix has zero rows")

    # Check design matrix
    if design is None:
        design = y.design
    if design is None:
        design = patsy.dmatrix("~1", data=y.exprs)
    else:
        if not isinstance(design, patsy.DesignMatrix):
            raise ValueError("design must be a patsy DesignMatrix")
        if design.shape[0] != y.exprs.shape[1]:
            raise ValueError(
                "row dimension of design does not match column dimension of data object"
            )
        if np.isnan(design).any():
            raise ValueError("NAs not allowed in design matrix")

    ne = nonEstimable(design)
    if ne is not None:
        logging.info(f"Coefficients not estimable: {' '.join(ne)}")

    # Check ndups and spacing. Default to 1.
    if ndups is None:
        try:
            ndups = y.printer.ndups
        except AttributeError:
            pass
    if ndups is None:
        ndups = 1
    if spacing is None:
        try:
            spacing = y.printer.spacing
        except AttributeError:
            pass
    if spacing is None:
        spacing = 1

    # Check weights
    if weights is None:
        weights = y.weights

    # Check method
    if method not in ["ls", "robust"]:
        raise ValueError(f"unknown method: {method}")

    # If duplicates are present, reduce probe-annotation and Amean to correct length
    if ndups > 1:
        if y.probes is not None:
            y.probes = uniquegenelist(y.probes, ndups=ndups, spacing=spacing)
        if y.Amean is not None:
            y.Amean = np.nanmean(
                unwrapdups(y.Amean, ndups=ndups, spacing=spacing), axis=1
            )

    # Dispatch fitting algorithms
    if method == "robust":
        raise NotImplementedError("robust method for lmFit is not implemented yet")
        # fit = mrlm(
        #    y.exprs, design=design, ndups=ndups, spacing=spacing, weights=weights
        # )
    else:
        if ndups < 2 and block is None:
            fit = lm_series(
                y.exprs, design=design, ndups=ndups, spacing=spacing, weights=weights
            )
        else:
            if correlation is None:
                raise ValueError(
                    "the correlation must be set, see function duplicateCorrelation"
                )
            raise NotImplementedError("gls_series is not implemented yet")
            # fit = gls_series(
            #    y.exprs,
            #    design=design,
            #    ndups=ndups,
            #    spacing=spacing,
            #    block=block,
            #    correlation=correlation,
            #    weights=weights,
            # )

    # Possible warning on missing coefs
    if fit.coefficients.shape[1] > 1:
        n = np.sum(np.isnan(fit.coefficients), axis=1)
        n = np.sum((n > 0) & (n < fit.coefficients.shape[1]))
        if n > 0:
            logging.warnings.warn(f"Partial NA coefficients for {n} probes")

    # output
    fit.genes = y.probes
    fit.Amean = y.Amean
    fit.method = method
    fit.design = design
    return fit


def lm_series(M, design=None, ndups=1, spacing=1, weights=None):
    """
    Fit linear model to microarray data by ordinary least squares

    Fit a linear model genewise to expression data from a series of arrays.
    This function uses ordinary least squares and is a utility function for
    :func:`lmFit`. Most users should not use this function directly but should
    use :func:`lmFit` instead.

    The linear model is fit for each gene by calling the function
    :func:`~inmoose.utils.lm_fit` or :func:`~inmoose.utils.lm_wfit`.

    Arguments
    ---------
    M : pd.DataFrame
        numeric matrix containing log-ratio or log-expression values for a
        series of microarrays, rows correspond to genes and columns to arrays
    design
        design matrix defining the linear model. The number of rows should
        agree with the number of columns of M. The number of columns will
        determine the number of coefficients estimated for each gene.
    ndups : int
        number of duplicate spots. Each gene is printed :code:`ndups` times in
        adjacent spots on each array
    spacing : int
        the spacing between the rows of :code:`M` corresponding to duplicate
        spots, :code:`spacing=1` for consecutive spots.
    weights : array_like
        an optional matrix of the same dimension as :code:`M` containing
        weights for each spot. If it is of different dimension to :code:`M`, it
        will be filled out to the same size.

    Returns
    -------
    MArrayLM
        an object with attributes

        - coefficients: matrix containing the estimated coefficients for each
          linear model. Same number of rows as :code:`M`, same number of
          columns as :code:`design`.
        - stdev_unscaled: matrix conformal with :code:`coef` containing the
          unscaled standard deviations for the coefficent estimators. The
          standard errors are given by :code:`stdev_unscaled * sigma`.
        - sigma: numeric vector containing the residual standard deviation for
          each gene.
        - df_residual: vector giving the degrees of freedom corresponding to
          :code:`sigma`.
        - qr: QR-decomposition of :code:`design`

    """
    # Check design
    if design is None:
        design = patsy.dmatrix("~1", data=M)

    nbeta = design.shape[1]
    try:
        coef_names = design.design_info.column_names
    except AttributeError:
        coef_names = [f"x{i}" for i in range(nbeta)]

    # Check weights
    if weights is not None:
        weights = np.broadcast_to(weights, M.shape)
        weights[weights <= 0] = np.nan
        M[~np.isfinite(weights)] = np.nan

    # Reform duplicated rows into columns
    if ndups > 1:
        M = unwrapdups(M, ndups=ndups, spacing=spacing)
        design = design @ np.arange(1, ndups + 1)
        if weights is not None:
            weights = unwrapdups(weights, ndups=ndups, spacing=spacing)

    # Initialize standard errors
    ngenes = M.shape[0]
    stdev_unscaled = pd.DataFrame(
        np.full((ngenes, nbeta), np.nan), columns=coef_names, index=M.index
    )
    beta = pd.DataFrame(
        np.full((ngenes, nbeta), np.nan), columns=coef_names, index=M.index
    )

    # if QR-decomposition is constant for all genes, fit all genes in one sweep
    NoProbesWts = np.all(np.isfinite(M)) and (weights is None)
    if NoProbesWts:
        if weights is None:
            fit = lm_fit(design, M.T)
        else:
            fit = lm_wfit(design, M.T, weights[0, :])
            fit.weights = None

        if fit.df_residuals > 0:
            if fit.effects.ndim == 2:
                sigma = np.sqrt(np.mean(fit.effects[fit.rank :, :] ** 2, axis=0))
            else:
                sigma = np.sqrt(np.mean(fit.effects[fit.rank :] ** 2))
        else:
            sigma = np.full(ngenes, np.nan)

        cov_coef = np.linalg.inv(design.T @ design)
        cov_coef = pd.DataFrame(cov_coef, index=coef_names, columns=coef_names)
        stdev_unscaled.loc[:, coef_names] = np.sqrt(np.diag(cov_coef))
        coef = fit.coefficients.T
        coef.index = M.index
        coef.columns = coef_names
        return MArrayLM(
            coef,
            stdev_unscaled,
            pd.Series(sigma, index=M.index),
            np.full(ngenes, fit.df_residuals),
            cov_coef,
        )

    # Genewise QR-decompositions are required, so iterate through genes
    sigma = np.full(ngenes, np.nan)
    df_residual = np.zeros(ngenes)
    for i in range(ngenes):
        y = M.iloc[i, :]
        obs = np.isfinite(y)
        if obs.sum() > 0:
            X = design[obs, :]
            y = y[obs]
            if weights is None:
                out = lm_fit(X, y)
            else:
                w = weights[i, obs]
                out = lm_wfit(X, y, w)

            est = ~np.isnan(out.coefficients)
            beta[i, :] = out.coefficients
            stdev_unscaled[i, est] = np.sqrt(np.diag(np.linalg.inv(X.T @ X)))
            df_residual[i] = out.df_residuals
            if df_residual[i] > 0:
                sigma[i] = np.sqrt(np.mean(out.effects[out.rank :] ** 2))

    # Correlation matrix of coefficients
    cov_coef = np.linalg.inv(design.T @ design)

    return MArrayLM(
        coefficients=beta,
        stdev_unscaled=stdev_unscaled,
        sigma=sigma,
        df_residual=df_residual,
        cov_coef=cov_coef,
    )


def nonEstimable(x):
    """
    Check whether a matrix has full column rank

    Returns a vector of names for the columns of :code:`x` which are linearly
    dependent on previous columns. If :code:`x` has full column rank, then
    returns :code:`None`.

    Arguments
    ---------
    x : array_like
        a design matrix

    Returns
    -------
    list of strings or None
        list of the names of the matrix columns which are linearly dependent on
        previous columns. :code:`None` if :code:`x` has full column rank
    """
    p = x.shape[1]
    Q, R, pivot = scipy.linalg.qr(x, pivoting=True)
    rank = np.linalg.matrix_rank(R)
    if rank < p:
        try:
            n = x.design_info.column_names
        except AttributeError:
            n = [f"{i}" for i in range(p)]
        n = np.array(n)
        notest = n[pivot[rank:]]
        blank = notest == ""
        if np.any(blank):
            notest[blank] = np.array([f"{i}" for i in range(rank, p)])[blank]
        return notest
    else:
        return None
