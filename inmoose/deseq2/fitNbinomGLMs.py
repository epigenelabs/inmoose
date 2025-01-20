# -----------------------------------------------------------------------------
# Copyright (C) 2013-2022 Michael I. Love, Constantin Ahlmann-Eltze
# Copyright (C) 2023-2024 Maximilien Colange

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

# This file is based on the file 'R/fitNbinomGLMs.R' of the Bioconductor DESeq2
# package (version 3.16).


from collections import OrderedDict

import numpy as np
import pandas as pd
import patsy
from scipy.optimize import Bounds, minimize

from ..utils import LOGGER, dnbinom_mu, dnorm
from .deseq2_cpp import fitBetaWrapper
from .misc import renameModelMatrixColumns
from .prior import estimateBetaPriorVar
from .weights import getAndCheckWeights


def nbinomLogLike(counts, mu, disp, weights, useWeights):
    """
    Compute the log likelihood for a count matrix, mu matrix and disp vector

    Arguments
    ---------
    counts : ndarray
        a count matrix
    mu : ndarray
        matrix of means. Should be broadcastable to the shape of :code:`counts`.
    disp : ndarray
        vector of dispersions. Should be broadcastable to the shape of :code:`counts`.
    weights : ndarray
        matrix of weights. Should be broadcastable to the shape of :code:`counts`.
    useWeights : bool
        whether to use weights
    """
    if disp is None:
        return np.full(counts.shape[1], np.nan)
    if useWeights:
        return np.sum(weights * dnbinom_mu(counts, mu=mu, size=1 / disp, log=True), 0)
    else:
        return np.sum(dnbinom_mu(counts, mu=mu, size=1 / disp, log=True), 0)


def fitNbinomGLMs(
    obj,
    modelMatrix=None,
    modelFormula=None,
    alpha_hat=None,
    lambda_=None,
    renameCols=True,
    betaTol=1e-8,
    maxit=100,
    useOptim=True,
    useQR=True,
    forceOptim=False,
    warnNonposVar=True,
    minmu=0.5,
    type_="DESeq2",
):
    """
    Fit negative binomial GLMs

    This is a low-level function. Users typically call :func:`nbinomWaldTest`
    or :func:`nbinomLRT` which calls this function to perform fitting. These
    functions return a :func:`DESeqDataSet` object with the appropriate columns
    added. This function returns its resuts as a list.

    Arguments
    ---------
    obj : DESeqDataSet
        a DESeqDataSet
    modelMatrix : ndarray
        the design matrix
    modelFormula
        a formula specifying how to construct the design matrix
    alpha_hat : array-like
        the dispersion parameters estimates
    lambda_ : array-like
        the "ridge" term added for the penalized GLM on the log2 scale
    renameCols : bool
        flag indicating whether to give columns variable_B_vs_A style names
    betaTol : float
        the relative tolerance for deviance. Fitting stops when
        :math:`abs(dev - dev_old) / (abs(dev) + 0.1) < betaTol`
    maxit : int
        the maximum number of iterations to allow for convergence
    useOptim : bool
        flag indicating whether to use optim on rows that have not converged.
        Fisher scoring is not ideal with multiple groups and sparse count
        distributions
    useQR : bool
        flag indicating whether to use the QR decomposition of the design matrix
    forceOptim : bool
        flag indicating whether to use optim on all rows
    warnNonposVar : bool
        flag indicating whether to warn about non-positive variances. This flag
        is intended for advanced users only running LRT without beta prior.
    minmu : float
        TODO
    type_ : str

    Returns
    -------
    """
    if type_ not in ["DESeq2", "glmGamPoi"]:
        raise ValueError(f"invalid value for type_: {type_}")

    if modelFormula is None:
        modelFormula = obj.design
    elif not isinstance(modelFormula, patsy.DesignMatrix):
        modelFormula = patsy.dmatrix(modelFormula, obj.obs)

    if modelMatrix is None:
        modelMatrix = obj.design

    if not np.all(np.sum(np.abs(modelMatrix), 0) > 0):
        raise ValueError("model matrix has 0 columns")

    # rename columns, for use as columns in DataFrame
    # and to emphasize the reference level comparison
    if renameCols:
        convertNames = renameModelMatrixColumns(obj.obs, modelFormula)
        modelMatrix.design_info.column_name_indexes = OrderedDict(
            [
                (convertNames[n] if n in convertNames else n, v)
                for n, v in modelMatrix.design_info.column_name_indexes.items()
            ]
        )

    modelMatrixNames = modelMatrix.design_info.column_names

    normalizationFactors = obj.getSizeOrNormFactors()

    if alpha_hat is None:
        alpha_hat = obj.var["dispersion"]

    if len(alpha_hat) != obj.n_vars:
        raise ValueError("alpha_hat needs to be the same length as obj.n_vars")

    # set a wide prior for all coefficients
    if lambda_ is None:
        lambda_ = np.repeat(1e-6, modelMatrix.shape[1])
    else:
        lambda_ = np.asarray(lambda_)
        lambda_ = lambda_.squeeze()
    assert lambda_.shape == (modelMatrix.shape[1],)

    # use weights if they are present in obj.layers
    (_, weights, useWeights) = getAndCheckWeights(obj, modelMatrix)

    if type_ == "glmGamPoi":
        raise NotImplementedError("glmGamPoi is not implemented")

    # bypass the beta fitting if the model formula is only intercept and
    # the prior variance is large (1e6)
    # i.e. LRT with reduced ~ 1 and no beta prior
    justIntercept = np.array_equal(
        [len(t.factors) for t in modelMatrix.design_info.terms], [0]
    )
    if justIntercept and np.all(lambda_ <= 1e-6):
        alpha = alpha_hat.values[None]
        betaConv = np.repeat(True, obj.n_vars)
        betaIter = np.ones(obj.n_vars)
        if useWeights:
            betaMatrix = np.log2(
                np.sum(weights * obj.counts(normalized=True), 0) / np.sum(weights, 0)
            )
        else:
            betaMatrix = np.log2(np.mean(obj.counts(normalized=True), 0))
        betaMatrix = pd.DataFrame(betaMatrix, columns=modelMatrixNames)
        mu = normalizationFactors * (2 ** betaMatrix.values.squeeze())
        logLikeMat = dnbinom_mu(obj.counts(), mu=mu, size=1 / alpha, log=True)
        if useWeights:
            logLike = np.sum(weights * logLikeMat, 0)
        else:
            logLike = np.sum(logLikeMat, 0)

        modelMatrix = patsy.dmatrix("~1", data=obj.obs)
        if useWeights:
            w = weights * 1 / (1 / mu + alpha)
        else:
            w = 1 / (1 / mu + alpha)

        xtwx = np.sum(w, 0)
        sigma = 1 / xtwx
        betaSE = pd.DataFrame(
            np.log2(np.exp(1) * np.sqrt(sigma)),
            columns=[f"SE_{n}" for n in modelMatrixNames],
        )
        hat_diagonals = w * sigma
        return {
            "logLike": logLike,
            "betaConv": betaConv,
            "betaMatrix": betaMatrix,
            "betaSE": betaSE,
            "mu": mu,
            "betaIter": betaIter,
            "modelMatrix": modelMatrix,
            "nterms": 1,
            "hat_diagonals": hat_diagonals,
        }

    # if full rank, estimate initial betas for IRLS below
    if np.linalg.matrix_rank(modelMatrix) == modelMatrix.shape[1]:
        q, r = np.linalg.qr(modelMatrix)
        y = np.log(obj.counts(normalized=True) + 0.1)
        beta_mat = np.linalg.solve(r, q.T @ y).T
    else:
        if patsy.Term([]) in modelMatrix.design_info.terms:
            beta_mat = np.zeros((obj.n_vars, modelMatrix.shape[1]))
            # use the natural log as fitBeta occurs in the natural log scale
            logBaseMean = np.log(np.mean(obj.counts(normalized=True), 0))
            beta_mat[:, modelMatrix.design_info.term_slices[patsy.Term([])]] = (
                logBaseMean[:, None]
            )
        else:
            beta_mat = np.ones((obj.n_vars, modelMatrix.shape[1]))

    # here we convert from the log2 scale of the betas
    # and the beta prior variance to the log scale used in fitBeta.
    # so we divide by the square of the conversion factor, log(2)
    lambdaNatLogScale = lambda_ / (np.log(2) ** 2)

    betaRes = fitBetaWrapper(
        y=obj.counts(),
        x=modelMatrix,
        nf=normalizationFactors,
        alpha_hat=alpha_hat.values,
        beta_mat=beta_mat,
        lambda_=lambdaNatLogScale,
        weights=weights,
        useWeights=useWeights,
        tol=betaTol,
        maxit=maxit,
        useQR=useQR,
        minmu=minmu,
    )

    # Note on deviance: the 'deviance' calculated in fitBeta()
    # is not returned in obj.var["deviance"]. Instead, we calculate
    # the log likelihood below and use -2 * logLike.
    # (reason is that we have other ways of estimating beta:
    # above intercept code, and below optim code)

    with np.errstate(over="ignore"):
        mu = normalizationFactors * np.exp(modelMatrix @ betaRes["beta_mat"].T)
    logLike = nbinomLogLike(
        obj.counts(),
        mu,
        obj.var["dispersion"].values if "dispersion" in obj.var else None,
        weights,
        useWeights,
    )

    # test for stability
    colStable = np.sum(np.isnan(betaRes["beta_mat"]), axis=1) == 0

    # test for positive variances
    colVarPositive = np.sum(betaRes["beta_var_mat"] <= 0, axis=1) == 0

    # test for convergence, stability and positive variances
    betaConv = betaRes["iter"] < maxit

    # here we transform the betaMatrix and betaSE to a log2 scale
    assert modelMatrix.design_info.column_names == modelMatrixNames
    betaMatrix = pd.DataFrame(
        np.log2(np.exp(1)) * betaRes["beta_mat"], columns=modelMatrixNames
    )
    # warn below regarding those rows with negative variance
    betaSE = pd.DataFrame(
        np.log2(np.exp(1)) * np.sqrt(np.maximum(betaRes["beta_var_mat"], 0)),
        columns=[f"SE_{n}" for n in modelMatrixNames],
    )

    if forceOptim:
        colsForOptim = np.arange(len(betaConv))
    else:
        # switch based on wether we should also use optim on rows which did not converge
        if useOptim:
            colsForOptim = ~betaConv | ~colStable | ~colVarPositive
        else:
            colsForOptim = ~colStable | ~colVarPositive
        colsForOptim = np.nonzero(colsForOptim)[0]

    if len(colsForOptim) > 0:
        assert betaMatrix.shape == beta_mat.shape, (
            f"{betaMatrix.shape} vs {beta_mat.shape}"
        )
        # we use optim if did not reach convergence with the IRLS code
        resOptim = fitNbinomGLMsOptim(
            obj,
            modelMatrix,
            lambda_,
            colsForOptim,
            colStable,
            normalizationFactors,
            alpha_hat,
            weights,
            useWeights,
            betaMatrix,
            betaSE,
            betaConv,
            beta_mat,
            mu,
            logLike,
            minmu=minmu,
        )
        betaMatrix = resOptim["betaMatrix"]
        betaSE = resOptim["betaSE"]
        betaConv = resOptim["betaConv"]
        mu = resOptim["mu"]
        logLike = resOptim["logLike"]

    if np.any(np.isnan(betaSE)):
        raise ValueError("betaSE contains NaN")
    nNonposVar = np.sum(np.sum(betaSE == 0, axis=0) > 0)
    if warnNonposVar and nNonposVar > 0:
        LOGGER.warning(
            f"{nNonposVar} cols had non-positive estimates of variance for coefficients"
        )

    return {
        "logLike": logLike,
        "betaConv": betaConv,
        "betaMatrix": betaMatrix,
        "betaSE": betaSE,
        "mu": mu,
        "betaIter": betaRes["iter"],
        "modelMatrix": modelMatrix,
        "nterms": modelMatrix.shape[1],
        "hat_diagonals": betaRes["hat_diagonals"],
    }


def fitGLMsWithPrior(
    obj, betaTol, maxit, useOptim, useQR, betaPriorVar, modelMatrix=None, minmu=0.5
):
    """this function call fitNbinomGLMs() twice:
    1. without the beta prior, in order to calculate the beta prior variance
       and hat matrix
    2. again but with the prior in order to get beta matrix and standard errors
    """
    objNZ = obj[:, ~obj.var["allZero"]]
    modelMatrixType = obj.modelMatrixType

    if betaPriorVar is None or not np.all(np.isin(["mu", "H"], objNZ.layers)):
        # stop unless modelMatrix was NOT supplied, the code below all works
        # by building model matrices using the formula, does not work with
        # incoming model matrices
        if modelMatrix is not None:
            raise ValueError()

        # fit the negative binomial GLM without a prior
        # used to construct the prior variances
        # and for the hat matrix diagonals for calculating Cook's distance
        fit = fitNbinomGLMs(
            objNZ,
            betaTol=betaTol,
            maxit=maxit,
            useOptim=useOptim,
            useQR=useQR,
            renameCols=(modelMatrixType == "standard"),
            minmu=minmu,
        )
        modelMatrix = fit["modelMatrix"]
        modelMatrixNames = modelMatrix.design_info.column_names
        H = fit["hat_diagonals"]
        betaMatrix = fit["betaMatrix"]
        mu = fit["mu"]

        betaMatrix.columns = modelMatrixNames

        # save the MLE log fold changes for addMLE argument of results
        convertNames = renameModelMatrixColumns(obj.obs, objNZ.design)
        modelMatrixNames = [
            convertNames[n] if n in convertNames else n for n in modelMatrixNames
        ]
        mleBetaMatrix = fit["betaMatrix"].copy()
        mleBetaMatrix.index = objNZ.var_names
        mleBetaMatrix.columns = [f"MLE_{n}" for n in modelMatrixNames]

        # store for use in estimateBetaPriorVar below
        objNZ.var = pd.concat([objNZ.var, mleBetaMatrix], axis=1)

    else:
        # we can skip the first MLE fit because the beta prior variance
        # and hat matrix diagonals were provided
        if modelMatrix is None:
            modelMatrix = obj.design

        H = objNZ.layers["H"]
        mu = objNZ.layers["mu"]
        mleBetaMatrix = objNZ.var.filter("MLE_")

    if betaPriorVar is None:
        betaPriorVar = estimateBetaPriorVar(objNZ, modelMatrix=modelMatrix)
    else:
        # else we are provided with the prior variance:
        # check if the lambda is the correct length given the design formula
        if modelMatrixType == "expanded":
            modelMatrix = objNZ.makeExpandedModelMatrix()

        p = modelMatrix.shape[1]
        if betaPriorVar.values.squeeze().shape[0] != p:
            raise ValueError(
                f"betaPriorVar should have length {p} to match {','.join(modelMatrix.design_info.column_names)}"
            )

    # refit the negative binomial GLM with a prior on beta
    if np.any(betaPriorVar == 0):
        raise ValueError("beta prior variances are zero for some variables")
    lambda_ = 1 / betaPriorVar.values.squeeze()
    assert lambda_.ndim < 2

    if modelMatrixType == "standard":
        fit = fitNbinomGLMs(
            objNZ,
            lambda_=lambda_,
            betaTol=betaTol,
            maxit=maxit,
            useOptim=useOptim,
            useQR=useQR,
            minmu=minmu,
        )
        modelMatrix = fit["modelMatrix"]
    elif modelMatrixType == "expanded":
        modelMatrix = objNZ.makeExpandedModelMatrix()
        fit = fitNbinomGLMs(
            objNZ,
            lambda_=lambda_,
            betaTol=betaTol,
            maxit=maxit,
            useOptim=useOptim,
            useQR=useQR,
            modelMatrix=modelMatrix,
            renameCols=False,
            minmu=minmu,
        )
    elif modelMatrixType == "user-supplied":
        fit = fitNbinomGLMs(
            objNZ,
            lambda_=lambda_,
            betaTol=betaTol,
            maxit=maxit,
            useOptim=useOptim,
            useQR=useQR,
            modelMatrix=modelMatrix,
            renameCols=False,
            minmu=minmu,
        )

    return {
        "fit": fit,
        "H": H,
        "betaPriorVar": betaPriorVar,
        "mu": mu,
        "modelMatrix": modelMatrix,
        "mleBetaMatrix": mleBetaMatrix,
    }


def fitNbinomGLMsOptim(
    obj,
    modelMatrix,
    lambda_,
    colsForOptim,
    colStable,
    normalizationFactors,
    alpha_hat,
    weights,
    useWeights,
    betaMatrix,
    betaSE,
    betaConv,
    beta_mat,
    mu,
    logLike,
    minmu=0.5,
):
    """breaking out the optim backup code from fitNbinomGLMs"""
    x = modelMatrix

    assert obj.n_obs == x.shape[0]
    assert normalizationFactors.shape == obj.shape
    assert alpha_hat.shape[0] == obj.n_vars
    assert not useWeights or obj.shape == weights.shape
    assert beta_mat.shape == (obj.n_vars, x.shape[1])
    assert betaMatrix.shape == betaSE.shape
    assert betaMatrix.shape == beta_mat.shape
    assert mu.shape == obj.shape
    assert lambda_.ndim == 1
    assert lambda_.shape[0] == x.shape[1], f"{lambda_.shape}, {x.shape}"
    assert colStable.shape[0] == obj.n_vars

    lambdaNatLogScale = lambda_ / np.log(2) ** 2
    large = 30
    for col in colsForOptim:
        if colStable[col] and np.all(np.abs(betaMatrix.iloc[col, :]) < large):
            betaCol = betaMatrix.iloc[col, :]
        else:
            betaCol = beta_mat[col, :]

        nf = normalizationFactors[:, col]
        k = obj.counts()[:, col]
        alpha = alpha_hat.iloc[col]

        def objectiveFn(p):
            with np.errstate(over="ignore"):
                mu_col = nf * 2 ** (x @ p)
            logLikeVector = dnbinom_mu(k, mu=mu_col, size=1 / alpha, log=True)
            if useWeights:
                logLike = np.sum(weights[:, col] * logLikeVector)
            else:
                logLike = np.sum(logLikeVector)
            logPrior = np.sum(dnorm(p, 0, np.sqrt(1 / lambda_), log=True))
            negLogPost = -(logLike + logPrior)
            if np.isfinite(negLogPost):
                return negLogPost
            else:
                return 1e300

        def jac(p):
            with np.errstate(over="ignore"):
                mu_col = nf * 2 ** (x @ p)
            if useWeights:
                res = -(weights[:, [col]] * x).T @ k + (
                    (1 / alpha + k) * mu_col / (1 / alpha + mu_col)
                ) @ (weights[:, [col]] * x)
            else:
                res = -x.T @ k + ((1 / alpha + k) * mu_col / (1 / alpha + mu_col)) @ x
            res /= np.log(2)
            res += lambda_ @ p
            return res

        o = minimize(
            objectiveFn,
            betaCol,
            method="TNC",
            bounds=Bounds(lb=-large, ub=large),
            jac=jac,
            tol=1e-6,
        )
        ridge = np.diag(lambdaNatLogScale)
        # if we converged, change betaConv to True
        if o.success:
            betaConv[col] = True

        # with or without convergence, store the estimate from optim
        betaMatrix.iloc[col, :] = o.x
        # calculate the standard errors
        with np.errstate(over="ignore"):
            mu_col = nf * 2 ** (x @ o.x)
        # store the new mu vector
        mu[:, col] = mu_col
        mu_col[mu_col < minmu] = minmu
        if useWeights:
            w = np.diag(weights[:, col] * 1 / (1 / mu_col + alpha))
        else:
            w = np.diag(1 / (1 / mu_col + alpha))

        xtwx = x.T @ w @ x
        xtwxRidgeInv = np.linalg.inv(xtwx + ridge)
        sigma = xtwxRidgeInv @ xtwx @ xtwxRidgeInv
        # warn below regarding those rows with negative variance
        betaSE.iloc[col, :] = np.log2(np.exp(1)) * np.sqrt(
            np.maximum(np.diag(sigma), 0)
        )
        logLikeVector = dnbinom_mu(k, mu=mu_col, size=1 / alpha, log=True)
        if useWeights:
            logLike[col] = np.sum(weights[:, col] * logLikeVector)
        else:
            logLike[col] = np.sum(logLikeVector)

    return {
        "betaMatrix": betaMatrix,
        "betaSE": betaSE,
        "betaConv": betaConv,
        "mu": mu,
        "logLike": logLike,
    }
