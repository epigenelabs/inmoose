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

# This file is based on the file 'R/core.R' of the Bioconductor DESeq2 package
# (version 3.16).


import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import polygamma
from scipy.stats import trim_mean
from statsmodels.tools.sm_exceptions import DomainWarning

from ..utils import LOGGER, Factor
from .deseq2_cpp import fitDispGridWrapper, fitDispWrapper
from .fitNbinomGLMs import fitNbinomGLMs
from .misc import buildMatrixWithNACols, buildVectorWithNACols, checkFullRank
from .weights import getAndCheckWeights


def estimateDispersions_dds(
    obj,
    fitType="parametric",
    maxit=100,
    useCR=True,
    weightThreshold=1e-2,
    quiet=False,
    modelMatrix=None,
    minmu=None,
):
    """
    Estimate the dispersions for a :class:`DESeqDataSet`

    This function obtains dispersion estimates for Negative Binomial
    distributed data. The function is typically called with the idiom:
    :code:`dds = dds.estimateDispersions()`.

    The fitting proceeds as follows: for each gene, an estimate of the
    dispersion is found which maximizes the Cox-Reid adjusted profile
    likelihood (the methods of Cox-Reid adjusted profile likelihood
    maximization for estimation of dispersion in RNA-Seq data were developed by
    [McCarthy2012]_, first implemented in the edgeR package in 2010 (see
    :func:`inmoose.edgepy.glmFit` and :func:`inmoose.edgepy.dispCoxReid`); a
    trend line capturing the dispersion-mean relationship is fit to the maximum
    likelihood estimates; a normal prior is determined for the log dispersion
    estimates centered on the predicted value from the trended fit with
    variance equal to the difference between the observed variance of the log
    dispersion estimates and te expected sampling variance; finally maximum a
    posteriori dispersion estimates are returned.  This final dispersion
    parameter is used in subsequent tests. The final dispersion estimates can
    be accessed through :attr:`DESeqDataSet.dispersions`. The fitted
    dispersion-mean relationship is also used in
    :func:`varianceStabilizingTransformation`. All of the intermediate values
    (gene-wise dispersion estimates, fitted dispersion estimates from the
    trended fit, etc.) are stored in :code:`dds.var`.

    The log normal prior on the dispersion parameter has been proposed by [Wu2012]_.

    See also
    --------
    ~inmoose.deseq2.estimateDispersionsGeneEst
        lower-level function called by this function
    ~inmoose.deseq2.estimateDispersionsFit
        lower-level function called by this function
    ~inmoose.deseq2.estimateDispersionsMAP
        lower-level function called by this function

    Arguments
    ---------
    obj : DESeqDataSet
        the input dataset
    fitType : "parametric", "local", "mean" or "glmGamPoi"
        the type of fitting the dispersions to the mean intensity.

        :code:`"parametric"` - fit a dispersion-mean relation of the form:
        :math:`dispersion = asymtDisp + extraPois / mean` *via* a robust
        gamma-family GLM.  The coefficients :code:`asymtDisp` and
        :code:`extraPois` are given in the attribute :code:`coefficients` of
        the :attr:`DESeqDataSet.dispersionFunction`.

        :code:`"local"` - use the locfit package to fit a local regression of
        log dispersions over log base mean (normal scale means and dispersions
        are input and output for :attr:`DESeqDataSet.dispersionFunction`. The
        points are weighted by normalized mean count in the local regression.

        :code:`"mean"` - use the mean of gene-wise dispersion estimates.

        :code:`"glmGamPoi"` - use the glmGamPoi package to fit the gene-wise
        dispersion, its trend, and calculate the MAP based on the
        quasi-likelihood framework. The trend is calculated using a local
        median regression.

    maxit : int
        maximum number of iterations to allow for convergence
    useCR : bool
        whether to use Cox-Reid adjustment (see [McCarthy2012]_)
    weightThreshold : float
        threshold for subsetting the design matrix and GLM weights for
        calculating the Cox-Reid correction
    quiet : bool
        whether to print messages at each step
    modelMatrix : array-like, optional
        an optional matrix which will be used for fitting the expected counts.
        By default, the model matrix is constructed from
        :attr:`DESeqDataSet.design`.
    minmu : float
        lower bound on the estimated count for fitting gene-wise dispersion

    Returns
    -------
    DESeqDataSet
        the input :code:`obj`, with the dispersion information filled in fields
        in :attr:`DESeqDataSet.var`, or the final dispersions accessible via
        :attr:`DESeqDataSet.dispersions`.
    """

    if fitType not in ["parametric", "local", "mean", "glmGamPoi"]:
        raise ValueError(f"invalid value for fitType: {fitType}")
    if minmu is None:
        if fitType == "glmGamPoi":
            minmu = 1e-6
        else:
            minmu = 0.5

    if obj.sizeFactors is None and obj.normalizationFactors is None:
        raise ValueError(
            "first call estimateSizeFactors or provide a normalizationFactor matrix before calling estimateDispersions"
        )

    # size factors could have slipped in the obs from a previous run
    if obj.sizeFactors is not None:
        if not np.issubdtype(obj.sizeFactors.dtype, np.number):
            raise ValueError(
                "the sizeFactor column in obs is not numeric. This column could have come in during obs import and should be removed."
            )
        if np.isnan(obj.sizeFactors).any():
            raise ValueError(
                "the sizeFactor column in obs contains NA. This column could have come in during obs import and should be removed."
            )

    if (np.sum(obj.X == obj.X[0], 1) == obj.n_obs).all():
        raise ValueError(
            "all genes have equal values for all samples. will not be able to perform differential analysis"
        )

    if "dispersion" in obj.var:
        LOGGER.info("found already estimated dispersions, replacing these")
        del obj.var["dispersion"]

    if fitType == "glmGamPoi":
        dispersionEstimator = "glmGamPoi"
        raise NotImplementedError(
            "estimating dispersions with glmGamPoi is not implemented"
        )
    else:
        dispersionEstimator = "DESeq2"

    checkForExperimentalReplicates(obj, modelMatrix)

    LOGGER.info("gene-wise dispersion estimates")
    obj = estimateDispersionsGeneEst(
        obj,
        maxit=maxit,
        useCR=useCR,
        weightThreshold=weightThreshold,
        quiet=quiet,
        modelMatrix=modelMatrix,
        minmu=minmu,
        type_=dispersionEstimator,
    )
    LOGGER.info("mean-dispersion relationship")
    obj = estimateDispersionsFit(obj, fitType=fitType, quiet=quiet)
    LOGGER.info("final dispersion estimates")
    obj = estimateDispersionsMAP(
        obj,
        maxit=maxit,
        useCR=useCR,
        weightThreshold=weightThreshold,
        quiet=quiet,
        modelMatrix=modelMatrix,
        type_=dispersionEstimator,
    )
    return obj


def estimateDispersionsGeneEst(
    obj,
    minDisp=1e-8,
    kappa_0=1,
    dispTol=1e-6,
    maxit=100,
    useCR=True,
    weightThreshold=1e-2,
    quiet=False,
    modelMatrix=None,
    niter=1,
    linearMu=None,
    minmu=None,
    alphaInit=None,
    type_="DESeq2",
):
    """
    Low-level function to fit dispersion estimates

    Normal users should instead use :meth:`.DESeqDataSet.estimateDispersions`.
    This low-level function is called by
    :meth:`.DESeqDataSet.estimateDispersions`, but is exported and documented
    for non-standard usage.  For instance, it is possible to replace fitted
    values with a custom fit and continue with the maximum a posteriori
    dispersion estimate.

    Arguments
    ---------
    obj : DESeqDataSet
        the input dataset
    minDisp : float
        small value for the minimum dispersion, to allow for calculations in
        log scale, one order of magnitude above this value is used as a test
        for inclusion in mean-dispersion fitting
    kappa_0 : float
        parameter used in setting the initial proposal in backtracking search,
        higher :code:`kappa_0` results in larger steps
    dispTol : float
        parameter to test for convergence of log dispersion, stop when increase
        in log posterior is less than :code:`dispTol`
    maxit : int
        maximum number of iterations to allow for convergence
    useCR : bool
        whether to use Cox-Reid adjustment
    weightThreshold : float
        threshold for subsetting the design matrix and GLM weights for
        calculating the Cox-Reid correction
    quiet : bool
        whether to print messages at each step
    modelMatrix : array-like
        for advanced use only, a substitute model matrix for gene-wise and MAP
        dispersion estimation
    niter : int
        number of times to iterate between estimation of means and estimation
        of dispersion
    linearMu :
        estimate the expected counts matrix using a linear model.
        Defaults to :code:`None`, in which case a linear model is used if the
        number of groups defined by the model matrix is equal to the number of
        columns of the model matrix.
    minmu : float
        lower bound on the estimated count for fitting gene-wise dispersion
    alphaInit : array-like
        initial guess for the dispersion estimates
    type_ : "DESeq2" or "glmGamPoi"
        specify if the glmGamPoi package is used to calculate the dispersion.
        This can be significantly faster if there are many replicates with
        small counts.

    Returns
    -------
    DESeqDataSet
        the input :code:`obj` with gene-wise dispersion estimates
    """
    if type_ not in ["DESeq2", "glmGamPoi"]:
        raise ValueError(f"invalid value for type_: {type_}")
    if minmu is None:
        if type_ == "glmGamPoi":
            minmu = 1e-6
        else:
            minmu = 0.5

    if "dispGeneEst" in obj.var:
        LOGGER.info("found already estimated gene-wise dispersions, removing these")
        del obj.var["dispGeneEst"]
        if "dispGeneIter" in obj.var:
            del obj.var["dispGeneIter"]

    if np.log(minDisp / 10) <= -30:
        raise ValueError(
            "for computational stability, log(minDisp/10) should be above -30"
        )

    if modelMatrix is None:
        modelMatrix = obj.design
    checkFullRank(modelMatrix)
    if modelMatrix.shape[0] == modelMatrix.shape[1]:
        raise ValueError(
            "the number of samples and the number of model coefficients are equal, i.e., there are no replicates to estimate the dispersion. Use an alternate design formula."
        )

    obj = obj.getBaseMeansAndVariances()

    # use weights if they are present
    # (we need this already to decide about linear mu fitting)
    obj.weightsOK = None
    (obj, weights, useWeights) = getAndCheckWeights(
        obj, modelMatrix, weightThreshold=weightThreshold
    )
    # don't let weights go below 1e-6
    weights = np.clip(weights, 1e-6, None)

    # only continue on the columns with non-zero mean
    objNZ = obj[:, ~obj.var["allZero"]]
    weights = weights[:, ~obj.var["allZero"]]

    if alphaInit is None:
        # this rough dispersion estimate (alpha_hat)
        # is for estimating mu
        # and for the initial starting point for line search
        roughDisp = roughDispEstimate(y=objNZ.counts(normalized=True), x=modelMatrix)
        momentsDisp = momentsDispEstimate(objNZ)
        alpha_hat = np.minimum(roughDisp, momentsDisp)
    else:
        if np.isscalar(alphaInit):
            alpha_hat = np.repeat(alphaInit, objNZ.n_var)
        else:
            if len(alphaInit) != objNZ.n_var:
                raise ValueError("len(alphaInit) and objNZ.n_var mismatch")
            alpha_hat = alphaInit

    # bound the rough estimated alpha between minDisp and maxDisp for numeric stability
    maxDisp = np.maximum(10, obj.n_obs)
    alpha_init = np.clip(alpha_hat, minDisp, maxDisp)
    alpha_hat_new = alpha_init.copy()
    alpha_hat = alpha_init.copy()

    if niter <= 0:
        raise ValueError("niter should be strictly positive")

    # use a linear model to estimate the expected counts
    # if the number of groups according to the model matrix
    # is equal to the number of columns
    if linearMu is None:
        modelMatrixGroups = Factor(
            [tuple(modelMatrix[i]) for i in range(modelMatrix.shape[0])]
        )
        linearMu = modelMatrixGroups.nlevels() == modelMatrix.shape[1]
        # also check for weights (then can't do linear mu)
        if useWeights:
            linearMu = False

    # below, iterate between mean and dispersion estimation (niter) times
    fitidx = np.repeat(True, objNZ.n_vars)
    mu = np.zeros(objNZ.shape)
    dispIter = np.zeros(objNZ.n_vars)
    # bound the estimated count by 'minmu'
    # this helps make the fitting more robust
    # because 1/mu occurs in the weights for the NB GLM
    for iter in range(niter):
        if not linearMu:
            fitMu = fitNbinomGLMs(
                objNZ[:, fitidx],
                alpha_hat=alpha_hat[fitidx],
                modelMatrix=modelMatrix,
                type_=type_,
            )["mu"]
        else:
            fitMu = linearModelMuNormalized(objNZ[:, fitidx], modelMatrix)

        fitMu[fitMu < minmu] = minmu
        mu[:, fitidx] = fitMu

        # use of kappa_0 in backtracking search
        # inital proposal = log(alpha) + kappa_0 * deriv. of log lik. w.r.t. log(alpha)
        # use log(minDisp/10) to stop if dispersions going to -infinity
        if type_ == "DESeq2":
            dispRes = fitDispWrapper(
                y=objNZ.counts()[:, fitidx],
                x=modelMatrix,
                mu_hat=fitMu,
                log_alpha=np.log(alpha_hat)[fitidx],
                log_alpha_prior_mean=np.log(alpha_hat)[fitidx],
                log_alpha_prior_sigmasq=1,
                min_log_alpha=np.log(minDisp / 10),
                kappa_0=kappa_0,
                tol=dispTol,
                maxit=maxit,
                usePrior=False,
                weights=weights[:, fitidx],
                useWeights=useWeights,
                weightThreshold=weightThreshold,
                useCR=useCR,
            )

            dispIter[fitidx] = dispRes["iter"]
            alpha_hat_new[fitidx] = np.minimum(np.exp(dispRes["log_alpha"]), maxDisp)
            last_lp = dispRes["last_lp"]
            initial_lp = dispRes["initial_lp"]
            # only rerun those cols which moved

        elif type_ == "glmGamPoi":
            raise NotImplementedError("glmGamPoi not implemented")

        fitidx = np.abs(np.log(alpha_hat_new) - np.log(alpha_hat)) > 0.5
        alpha_hat = alpha_hat_new
        if np.sum(fitidx) == 0:
            break

    # dont accept moves if the log posterior did not
    # increase by more than one millionth,
    # and set the small estimates to the minimum dispersion
    dispGeneEst = alpha_hat
    if niter == 1:
        noIncrease = last_lp < initial_lp + np.abs(initial_lp) / 1e6
        dispGeneEst[noIncrease] = alpha_init[noIncrease]
    # did not reach the maximum and iterated more than once
    dispGeneEstConv = (dispIter < maxit) & (dispIter > 1)

    # if lacking convergence from fitDisp() (C++)...
    refitDisp = ~dispGeneEstConv & (dispGeneEst > minDisp * 10)
    if np.sum(refitDisp) > 0:
        dispGrid = fitDispGridWrapper(
            y=objNZ.counts()[:, refitDisp],
            x=modelMatrix,
            mu=mu[:, refitDisp],
            log_alpha_prior_mean=np.repeat(0, np.sum(refitDisp)),
            log_alpha_prior_sigmasq=1,
            usePrior=False,
            weights=weights[:, refitDisp],
            useWeights=useWeights,
            weightThreshold=weightThreshold,
            useCR=useCR,
        )
        dispGeneEst[refitDisp] = dispGrid

    dispGeneEst = np.clip(dispGeneEst, minDisp, maxDisp)

    obj.var["dispGeneEst"] = buildVectorWithNACols(dispGeneEst, obj.var["allZero"])
    obj.var.type["dispGeneEst"] = "intermediate"
    obj.var.description["dispGeneEst"] = "gene-wise estimates of dispersion"
    obj.var["dispGeneIter"] = buildVectorWithNACols(dispIter, obj.var["allZero"])
    obj.var.type["dispGeneIter"] = "intermediate"
    obj.var.description["dispGeneIter"] = "number of iterations for gene-wise"

    obj.layers["mu"] = buildMatrixWithNACols(mu, obj.var["allZero"])

    return obj


def estimateDispersionsFit(obj, fitType="parametric", minDisp=1e-8, quiet=False):
    """
    Low-level function to fit dispersion estimates

    Normal users should instead use :meth:`.DESeqDataSet.estimateDispersions`.
    This low-level function is called by
    :meth:`.DESeqDataSet.estimateDispersions`, but is exported and documented
    for non-standard usage.  For instance, it is possible to replace fitted
    values with a custom fit and continue with the maximum a posteriori
    dispersion estimate.

    Arguments
    ---------
    obj : DESeqDataSet
        the input dataset
    fitType : "parametric", "local", "mean" or "glmGamPoi"
        the type of fitting of dispersions to the mean intensity.
        See :meth:`.DESeqDataSet.estimateDispersions` for description.
    minDisp : float
        small value for the minimum dispersion, to allow for calculations in
        log scale, one order of magnitude above this value is used as a test
        for inclusion in mean-dispersion fitting
    quiet : bool
        whether to print messages at each step

    Returns
    -------
    DESeqDataSet
        the input :code:`obj` with fitted dispersion estimates
    """
    if "allZero" not in obj.var:
        obj = obj.getBaseMeansAndVariances()

    objNZ = obj[:, ~obj.var["allZero"]]
    useForFit = objNZ.var["dispGeneEst"] > 100 * minDisp
    if useForFit.sum() == 0:
        raise ValueError(
            """
            all gene-wise dispersion estimates are within 2 orders of magnitude
            from the minimum value, and so the standard curve fitting techniques will not work.
            One can instead use the gene-wise estimates as final estimates:
            dds = estimateDispersionsGeneEst(dds)
            dds.dispersions = dds.var["dispGeneEst"]
            ... then continue with testing using nbinonWaldTest or nbinomLRT
            """
        )

    if fitType not in ["parametric", "local", "mean", "glmGamPoi"]:
        raise ValueError(f"incorrect value for fitType: {fitType}")

    if fitType == "parametric":
        try:
            dispFunction = parametricDispersionFit(
                objNZ.var["baseMean"][useForFit], objNZ.var["dispGeneEst"][useForFit]
            )
        except RuntimeError as e:
            LOGGER.info(f"parametric fit failed with {e}")
            LOGGER.info(
                "note: fitType='parametric', but the dispersion trend was not well captured by the function: y = a/x + b, and a local regression fit was automatically substituted. Specify fitType='local' or 'mean' to avoid this message next time."
            )
            LOGGER.info(
                "NB: original DESeq2 goes for a local regression fit, which is currently not implemented in inmoose. Specify fitType='mean' to avoid this."
            )
            fitType = "local"

    if fitType == "local":
        raise NotImplementedError()
        # dispFunction = localDispersionFit(
        # means = objNZ.var["baseMean"][useForFit],
        # disps = objNZ.var["dispGeneEst"][useForFit],
        # minDisp = minDisp)

    if fitType == "mean":
        useForMean = objNZ.var["dispGeneEst"] > 10 * minDisp
        useForMean = useForMean & ~np.isnan(objNZ.var["dispGeneEst"])
        meanDisp = trim_mean(objNZ.var["dispGeneEst"][useForMean], 0.001)

        def dispFunction(means):
            return meanDisp

        dispFunction.mean = meanDisp

    if fitType == "glmGamPoi":
        raise NotImplementedError()

    # store the dispersion function and attributes
    dispFunction.fitType = fitType
    obj.setDispFunction(dispFunction)

    return obj


def estimateDispersionsMAP(
    obj,
    outlierSD=2,
    dispPriorVar=None,
    minDisp=1e-8,
    kappa_0=1,
    dispTol=1e-6,
    maxit=100,
    useCR=True,
    weightThreshold=1e-2,
    modelMatrix=None,
    type_="DESeq2",
    quiet=False,
):
    """
    Low-level function to fit dispersion estimates

    Normal users should instead use :meth:`.DESeqDataSet.estimateDispersions`.
    This low-level function is called by
    :meth:`.DESeqDataSet.estimateDispersions`, but is exported and documented
    for non-standard usage.  For instance, it is possible to replace fitted
    values with a custom fit and continue with the maximum a posteriori
    dispersion estimate.

    :func:`estimateDispersionsPriorVar` is called inside this function, and
    stores the dispersion prior variance as an attribute of
    :attr:`.DESeqDataSet.dispersionFunction`, which can be manually provided to
    :code:`estimateDispersionsMAP` for parallel execution.

    Arguments
    ---------
    obj : DESeqDataSet
        the input dataset
    outlierSD : int
        the number of standard deviations of log gene-wise estimates above the
        prior mean (fitted value), above which dispersion estimates will be
        labelled outliers. Outliers will keep their original value and not be
        shrunk using the prior.
    dispPriorVar : array-like
        the variance of the normal priori on the log dispersions. If not
        supplied, it is calculated as the difference between the mean squared
        residuals of gene-wise estimates to the fitted dispersion and the
        expected sampling variance of the log dispersion.
    minDisp : float
        small value for the minimum dispersion, to allow for calculations in
        log scale, one order of magnitude above this value is used as a test
        for inclusion in mean-dispersion fitting
    kappa_0 : float
        parameter used in setting the initial proposal in backtracking search,
        higher :code:`kappa_0` results in larger steps
    dispTol : float
        parameter to test for convergence of log dispersion, stop when increase
        in log posterior is less than :code:`dispTol`
    maxit : int
        maximum number of iterations to allow for convergence
    useCR : bool
        whether to use Cox-Reid adjustment
    weightThreshold : float
        threshold for subsetting the design matrix and GLM weights for
        calculating the Cox-Reid correction
    quiet : bool
        whether to print messages at each step
    modelMatrix : array-like
        for advanced use only, a substitute model matrix for gene-wise and MAP
        dispersion estimation
    type_ : "DESeq2" or "glmGamPoi"
        specify if the glmGamPoi package is used to calculate the dispersion.
        This can be significantly faster if there are many replicates with
        small counts.

    Returns
    -------
    DESeqDataSet
        the input :code:`obj` with final MAP dispersion estimates
    """
    if type_ not in ["DESeq2", "glmGamPoi"]:
        raise ValueError(f"invalid value for type_: {type_}")

    if "allZero" not in obj.var:
        obj = obj.getBaseMeansAndVariances()
    if "dispersion" in obj.var:
        LOGGER.info("found already estimated dispersions, removing these")
        del obj.var["dispersion"]
        del obj.var["dispOutlier"]
        del obj.var["dispMAP"]
        del obj.var["dispIter"]
        del obj.var["dispConv"]

    if modelMatrix is None:
        modelMatrix = obj.design

    # fill in the calculated dispersion prior variance
    if dispPriorVar is None:
        # if no gene-wise estimates above minimum
        if np.nansum(obj.var["dispGeneEst"] >= 100 * minDisp) == 0:
            LOGGER.warning(
                f"all genes have dispersion estimates < {100 * minDisp}, returning disp = {10 * minDisp}"
            )
            obj.var["dispersion"] = buildVectorWithNACols(
                10 * minDisp, obj.var["allZero"]
            )
            obj.var.type["dispersion"] = "intermediate"
            obj.var.description["dispersion"] = "final estimates of dispersion"
            dispFn = obj.dispersionFunction
            dispFn.dispPriorVar = 0.25
            obj.setDispFunction(dispFn, estimateVar=False)
            return obj

        dispPriorVar = estimateDispersionsPriorVar(obj, modelMatrix=modelMatrix)
        dispFn = obj.dispersionFunction
        dispFn.dispPriorVar = dispPriorVar
        obj.setDispFunction(dispFn, estimateVar=False)

    # use weights if they are present in obj.layers
    (obj, weights, useWeights) = getAndCheckWeights(
        obj, modelMatrix, weightThreshold=weightThreshold
    )

    objNZ = obj[:, ~obj.var["allZero"]]
    weights = weights[:, ~obj.var["allZero"]]
    varLogDispEsts = obj.dispersionFunction.varLogDispEsts

    # set prior variance for fitting dispersion
    log_alpha_prior_sigmasq = dispPriorVar

    # get previously calculated mu
    mu = objNZ.layers["mu"]

    if type_ == "DESeq2":
        # start fitting at gene estimate unless the points are one order of magnitude
        # below the fitted line, then start at fitted line
        dispInit = np.where(
            objNZ.var["dispGeneEst"] > 0.1 * objNZ.var["dispFit"],
            objNZ.var["dispGeneEst"],
            objNZ.var["dispFit"],
        )

        # if any missing values, fill in the fitted values to initialize
        dispInit[np.isnan(dispInit)] = objNZ.var["dispFit"][np.isnan(dispInit)]

        # run with prior
        dispResMAP = fitDispWrapper(
            y=objNZ.counts(),
            x=modelMatrix,
            mu_hat=mu,
            log_alpha=np.log(dispInit),
            log_alpha_prior_mean=np.log(objNZ.var["dispFit"]),
            log_alpha_prior_sigmasq=log_alpha_prior_sigmasq,
            min_log_alpha=np.log(minDisp / 10),
            kappa_0=kappa_0,
            tol=dispTol,
            maxit=maxit,
            usePrior=True,
            weights=weights,
            useWeights=useWeights,
            weightThreshold=weightThreshold,
            useCR=useCR,
        )

        # prepare dispersions for storage
        dispMAP = np.exp(dispResMAP["log_alpha"])
        dispIter = dispResMAP["iter"]

        # when lacking convergence from fitDisp()
        # we use a function to maximize dispersion parameter
        # along an adaptive grid
        dispConv = dispResMAP["iter"] < maxit
        refitDisp = ~dispConv
        if np.sum(refitDisp) > 0:
            dispGrid = fitDispGridWrapper(
                y=objNZ.counts()[:, refitDisp],
                x=modelMatrix,
                mu=mu[:, refitDisp],
                log_alpha_prior_mean=np.log(objNZ.var["dispFit"])[refitDisp],
                log_alpha_prior_sigmasq=log_alpha_prior_sigmasq,
                usePrior=True,
                weights=weights[:, refitDisp],
                useWeights=useWeights,
                weightThreshold=weightThreshold,
                useCR=True,
            )
            dispMAP[refitDisp] = dispGrid
    elif type_ == "glmGamPoi":
        raise NotImplementedError()

    # bound the dispersion estimate between minDisp and maxDisp for numeric stability
    maxDisp = np.maximum(10, obj.n_obs)
    dispMAP = np.clip(dispMAP, minDisp, maxDisp)

    dispersionFinal = dispMAP.copy()

    # detect outliers which have gene-wise estimates
    # outlierSD * standard deviation of log gene-wise estimates
    # above the fitted mean (prior mean)
    # and keep the original gene-est value for these.
    # Note: we use the variance of log dispersions estimates
    # from all the genes, not only those from below
    dispOutlier = np.log(objNZ.var["dispGeneEst"]) > np.log(
        objNZ.var["dispFit"]
    ) + outlierSD * np.sqrt(varLogDispEsts)
    dispOutlier[np.isnan(dispOutlier)] = False
    dispersionFinal[dispOutlier] = objNZ.var["dispGeneEst"][dispOutlier]

    obj.var["dispersion"] = buildVectorWithNACols(dispersionFinal, obj.var["allZero"])
    obj.var.type["dispersion"] = "intermediate"
    obj.var.description["dispersion"] = "final estimates of dispersion"
    obj.var["dispIter"] = buildVectorWithNACols(dispIter, obj.var["allZero"])
    obj.var.type["dispIter"] = "intermediate"
    obj.var.description["dispIter"] = "number of iterations"
    obj.var["dispOutlier"] = buildVectorWithNACols(dispOutlier, obj.var["allZero"])
    obj.var.type["dispOutlier"] = "intermediate"
    obj.var.description["dispOutlier"] = "dispersion flagged as outlier"
    obj.var["dispMAP"] = buildVectorWithNACols(dispMAP, obj.var["allZero"])
    obj.var.type["dispMAP"] = "intermediate"
    obj.var.description["dispMAP"] = "maximum a posteriori estimate"

    return obj


def estimateDispersionsPriorVar(obj, minDisp=1e-8, modelMatrix=None):
    """
    Low-level function to fit dispersion estimates

    Normal users should instead use :meth:`.DESeqDataSet.estimateDispersions`.
    This low-level function is called by
    :meth:`.DESeqDataSet.estimateDispersions`, but is exported and documented
    for non-standard usage.  For instance, it is possible to replace fitted
    values with a custom fit and continue with the maximum a posteriori
    dispersion estimate.

    This function is called inside :func:`estimateDispersionsMAP`, and stores
    the dispersion prior variance as an attribute of
    :attr:`.DESeqDataSet.dispersionFunction`, which can be manually provided to
    :func:`estimateDispersionsMAP` for parallel execution.

    Arguments
    ---------
    obj : DESeqDataSet
        the input dataset
    minDisp : float
        small value for the minimum dispersion, to allow for calculations in
        log scale, one order of magnitude above this value is used as a test
        for inclusion in mean-dispersion fitting
    modelMatrix : array-like
        for advanced use only, a substitute model matrix for gene-wise and MAP
        dispersion estimation

    Returns
    -------
    DESeqDataSet
        the input :code:`obj` with final MAP dispersion estimates
    """
    objNZ = obj[:, ~obj.var["allZero"]]
    aboveMinDisp = objNZ.var["dispGeneEst"] >= 100 * minDisp
    if modelMatrix is None:
        modelMatrix = objNZ.design
    # estimate the variance of the distribution of the
    # log dispersion estimates around the fitted value
    # dispResiduals = np.log(objNZ.var["dispGeneEst"]) - np.log(objNZ.var["dispFit"])
    if np.nansum(aboveMinDisp) == 0:
        raise ValueError("no data found which is greater than minDisp")

    varLogDispEsts = obj.dispersionFunction.varLogDispEsts

    m, p = modelMatrix.shape

    # if the residual degrees of freedom is between 1 and 3, the distribution
    # of log dispersions is especially asymmetric and poorly estimated
    # by the MAD. We then use an alternate estimator, a Monte Carlo
    # approach to match the distribution.
    if (m - p) <= 3 and m > p:
        # TODO implement Monte Carlo re-estimation
        pass

    # estimate the expected sampling variance of the log estimates
    # Var(log(cX)) = Var(log(X))
    # X ~ chi-squared with m - p degrees of freedom
    if m > p:
        expVarLogDisp = polygamma(1, (m - p) / 2)
        # set the variance of the prior using these two estimates
        # with a minimum of 0.25
        dispPriorVar = np.maximum(varLogDispEsts - expVarLogDisp, 0.25)
    else:
        # we have m = p, so do not try to subtract sampling variance
        dispPriorVar = varLogDispEsts
        expVarLogDisp = 0

    return dispPriorVar


def checkForExperimentalReplicates(obj, modelMatrix):
    if modelMatrix is None:
        modelMatrix = obj.design

    if modelMatrix.shape[0] == modelMatrix.shape[1]:
        raise ValueError(
            """
                The design matrix has the same number of samples and coefficients to fit,
                so estimation of dispersion is not possible. Treating samples
                as replicates is not supported."""
        )


def roughDispEstimate(y, x):
    """rough dispersion estimate using counts and fitted values

    Arguments
    ---------
    y : array-like
        normalized counts matrix (shape nobs x nvar)
    x : array-like
        design matrix (shape nobs x nd)
    """
    # must be positive
    mu = linearModelMu(y, x)
    mu = np.clip(mu, 1, None)

    m, p = x.shape

    # an alternate rough estimator with higher mean squared or absolute error
    # (colSums( (y - mu)^2/(mu * (m - p)) ) - 1) / colMeans(mu)

    # rough disp estimates will be adjusted up to minDisp later
    est = np.sum(((y - mu) ** 2 - mu) / mu**2, 0) / (m - p)
    return np.clip(est, 0, None)


def linearModelMu(y, x):
    """
    Arguments
    ---------
    y : array-like
        counts matrix
    x : array-like
        design matrix (as many rows as y)
    """
    # NB: in the R version, y is transposed compared to this Python version
    # original R version: ((x Rinv Q.T) y.T).T
    # optimized R version: (y Q) (x Rinv).T
    # we choose to try (x Rinv) (Q.T y), which seems to minimize the size of intermediate matrices (assuming p <= nobs)

    # NB: the R code, and the code below, assumes that p <= nobs.
    # this is guaranteed by the way we check that the design matrix is full rank.
    (Q, R) = np.linalg.qr(x)
    Rinv = np.linalg.solve(R, np.identity(R.shape[0]))
    return (x @ Rinv) @ (Q.T @ y)


def linearModelMuNormalized(obj, x):
    """
    Arguments
    ---------
    y : DESeqDataSet
    x : array-like
        design matrix
    """
    norm_cts = obj.counts(normalized=True)
    muhat = linearModelMu(norm_cts, x)
    nf = obj.getSizeOrNormFactors()
    return muhat * nf


def momentsDispEstimate(obj):
    if obj.normalizationFactors is not None:
        xim = np.mean(1 / np.mean(obj.normalizationFactors, 1))
    else:
        xim = np.mean(1 / obj.sizeFactors)

    bv = obj.var["baseVar"]
    bm = obj.var["baseMean"]
    return (bv - xim * bm) / bm**2


def parametricDispersionFit(means, disps):
    """Estimate a parametric fit of dispersion to the mean intensity

    Arguments
    ---------
    means : ??
    disps : ??
    """
    warnings.simplefilter("ignore", DomainWarning)
    coefs = pd.Series([0.1, 1.0])
    iter_ = 0
    while True:
        residuals = disps / (coefs.iloc[0] + coefs.iloc[1] / means)
        good = (residuals > 1e-4) & (residuals < 15)
        # check for glm convergence below to exit while loop
        glm_gamma = sm.GLM(
            disps[good],
            sm.add_constant(1 / means[good]),
            family=sm.families.Gamma(link=sm.families.links.Identity()),
        )
        fit = glm_gamma.fit(start_params=coefs)
        oldcoefs = coefs.copy()
        coefs = fit.params

        if not np.all(coefs > 0):
            raise RuntimeError("parametric dispersion fit failed")
        if np.sum(np.log(coefs / oldcoefs.values) ** 2) < 1e-6 and fit.converged:
            break
        iter_ = iter_ + 1
        if iter_ > 10:
            raise RuntimeError("dispersion fit did not converge")

    coefs.index = ["asymptDisp", "extraPois"]

    def ans(q):
        return coefs.iloc[0] + coefs.iloc[1] / q

    ans.coefficients = coefs
    return ans
