# -----------------------------------------------------------------------------
# Copyright (C) 2013-2022 Michael I. Love, Constantin Ahlmann-Eltze
# Copyright (C) 2023 Maximilien Colange

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


import logging
import numpy as np
import pandas as pd
from scipy.stats import trim_mean

from ..utils import Factor, pnorm, pt
from .fitNbinomGLMs import fitNbinomGLMs, fitGLMsWithPrior
from .misc import buildMatrixWithNACols, nOrMoreInCell, buildDataFrameWithNACols
from .weights import getAndCheckWeights


def nbinomWaldTest(
    obj,
    betaPrior=False,
    betaPriorVar=None,
    modelMatrix=None,
    modelMatrixType=None,
    betaTol=1e-8,
    maxit=100,
    useOptim=True,
    quiet=False,
    useT=False,
    df=None,
    useQR=True,
    minmu=0.5,
):
    """\
    TODO

    Arguments
    ---------
    """
    if not quiet:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    if "dispersion" not in obj.var:
        raise ValueError(
            "testing requires dispersion estimates, first call estimateDispersions()"
        )

    if not obj.var.type.filter("results").empty:
        logging.info("found results columns, replacing these")
        obj = obj.removeResults()

    if obj.var["allZero"] is None:
        obj = obj.getBaseMeansAndVariances()

    # only continue on the columns with non-zero means
    objNZ = obj[:, ~obj.var["allZero"]]

    # model matrix not provided...
    if modelMatrix is None:
        modelAsFormula = True
        termsOrder = np.array([len(t.factors) for t in obj.design.design_info.terms])
        interactionPresent = (termsOrder > 1).any()

        # run some tests common to DESeq, nbinomWaldTest, nbinomLRT
        obj.designAndArgChecker(betaPrior)

        # what kind of model matrix to use
        if not isinstance(betaPrior, bool):
            raise ValueError("betaPrior must be a boolean")
        blindDesign = np.array_equal(termsOrder, [0])
        if blindDesign:
            betaPrior = False
        if modelMatrixType is None:
            if betaPrior:
                modelMatrixType = "expanded"
            else:
                modelMatrixType = "standard"
        if modelMatrixType == "expanded" and not betaPrior:
            raise ValueError("expanded model matrices require a beta prior")

        # store modelMatrixType so it can be accessed by estimateBetaPriorVar
        obj.modelMatrixType = modelMatrixType
        hasIntercept = 0 in termsOrder
        renameCols = hasIntercept
    else:  # modelMatrix is not None, user-supplied
        if betaPrior:
            if betaPriorVar is None:
                raise ValueError(
                    "user-supplied model matrix with betaPrior=True requires supplying betaPriorVar"
                )
        modelAsFormula = False
        obj.modelMatrixType = "user-supplied"
        renameCols = False

    if not betaPrior:
        # fit the negative binomial GLM without a prior
        # (in actuality a very wide prior with standard deviation 1e3 on log2 fold changes)
        fit = fitNbinomGLMs(
            objNZ,
            betaTol=betaTol,
            maxit=maxit,
            useOptim=useOptim,
            useQR=useQR,
            renameCols=renameCols,
            modelMatrix=modelMatrix,
            minmu=minmu,
        )
        H = fit["hat_diagonals"]
        mu = fit["mu"]
        modelMatrix = fit["modelMatrix"]
        # record the wide prior variance which was used in fitting
        betaPriorVar = np.full(modelMatrix.shape[1], 1e6)
    else:
        priorFitList = fitGLMsWithPrior(
            obj=obj,
            betaTol=betaTol,
            maxit=maxit,
            useOptim=useOptim,
            useQR=useQR,
            betaPriorVar=betaPriorVar,
            modelMatrix=modelMatrix,
            minmu=minmu,
        )
        fit = priorFitList["fit"]
        H = priorFitList["H"]
        mu = priorFitList["mu"]
        modelMatrix = priorFitList["modelMatrix"]
        betaPriorVar = priorFitList["betaPriorVar"]
        mleBetaMatrix = priorFitList["mleBetaMatrix"]

        # will add the MLE beta, so remove any which exist already
        # (possibly coming from estimateMLEForBetaPriorVar)
        MLEcols = obj.var.filter(regex="MLE_").columns
        for c in MLEcols:
            del obj.var[c]
            del obj.var.type[c]
            del obj.var.description[c]

    # store mu and H, the hat matrix diagonals
    objNZ.layers["mu"] = mu
    obj.layers["mu"] = buildMatrixWithNACols(mu, obj.var["allZero"])
    objNZ.layers["H"] = H
    obj.layers["H"] = buildMatrixWithNACols(H, obj.var["allZero"])

    # store the prior variance directly as an attribute of the DESeqDataSet
    # object, so it can be pulled later by the results functions
    # (necessary for setting max Cook's distance)
    obj.betaPrior = betaPrior
    obj.betaPriorVar = betaPriorVar
    obj.modelMatrix = modelMatrix
    obj.test = "Wald"

    # compute Cook's distance
    if modelAsFormula:
        dispModelMatrix = obj.design
    else:
        dispModelMatrix = modelMatrix
    obj.dispModelMatrix = dispModelMatrix
    cooks = calculateCooksDistance(objNZ, H, dispModelMatrix)

    # record maximum Cook's
    maxCooks = recordMaxCooks(obj.design, obj.obs, dispModelMatrix, cooks, objNZ.n_vars)

    # store Cook's distance for each sample
    obj.layers["cooks"] = buildMatrixWithNACols(cooks, obj.var["allZero"])

    # add betas, standard errors and Wald p-values to the object
    modelMatrixNames = modelMatrix.design_info.column_names
    betaMatrix = fit["betaMatrix"]
    assert isinstance(betaMatrix, pd.DataFrame)
    assert betaMatrix.shape == (
        objNZ.n_vars,
        len(modelMatrixNames),
    ), f"betaMatrix shape {betaMatrix.shape} is wrong, should be {(objNZ.n_vars, len(modelMatrixNames))}"
    betaMatrix.index = objNZ.var_names
    assert np.array_equal(betaMatrix.columns, modelMatrixNames)
    betaSE = fit["betaSE"]
    assert isinstance(betaSE, pd.DataFrame)
    assert (
        betaSE.shape == betaMatrix.shape
    ), f"betaSE and betaMatrix shapes disagree: {betaSE.shape} and {betaMatrix.shape}"
    betaSE.index = objNZ.var_names
    betaSE.columns = [f"SE_{n}" for n in modelMatrixNames]
    WaldStatistic = betaMatrix / betaSE.values
    WaldStatistic.columns = [f"WaldStatistic_{n}" for n in modelMatrixNames]

    #################################
    ## t distribution for p-values ##
    #################################

    if useT:
        # if the `df` was provided to nbinomWaldTest...
        if df is not None:
            df = np.asarray(df)
            if len(df) != obj.n_vars:
                raise ValueError(
                    "df should have the same length as the number of vars in obj"
                )
            if len(df.shape) == 0:
                df = np.repeat(df, objNZ.n_vars)
            else:
                # the WaldStatistic vector is along nonzero cols of obj
                df = df[~obj.var["allZero"]]
        else:
            # df is missing, so compute it from the number of samples (wrt weights)
            # and the number of coefficients
            if "weights" in obj.layers:
                # this checks that weights are OK and normalizes to have colMax == 1
                # (although this has already happened earlier in estDispGeneEst and estDispMAP...)
                (_, weights, _) = getAndCheckWeights(objNZ, dispModelMatrix)
                num_samps = np.sum(weights, axis=0)
            else:
                num_samps = np.repeat(obj.n_obs, objNZ.n_vars)

            df = num_samps - dispModelMatrix.shape[1]

        df = np.where(df > 0, df, np.nan)
        if df.shape[0] != WaldStatistic.shape[0]:
            raise RuntimeError(
                f"df and WaldStatistic have incompatible shapes: {df.shape} vs {WaldStatistic.shape}"
            )
        # use a t distribution to calculate the p-value
        WaldPvalue = 2 * pt(np.abs(WaldStatistic), df=df[:, None], lower_tail=False)
    else:
        WaldPvalue = 2 * pnorm(np.abs(WaldStatistic), lower_tail=False)

    WaldPvalue = pd.DataFrame(
        WaldPvalue,
        index=objNZ.var_names,
        columns=[f"WaldPvalue_{n}" for n in modelMatrixNames],
    )

    betaConv = fit["betaConv"]

    if np.any(~betaConv):
        logging.info(
            f"{np.sum(~betaConv)} cols did not converge in beta, labelled in obj.var['betaConv']. Use larger maxit argument with nbinomWaldTest"
        )

    resultsList = [betaMatrix, betaSE, WaldStatistic, WaldPvalue]
    if betaPrior:
        resultsList.append(mleBetaMatrix)
    resultsDF = pd.concat(resultsList, axis=1)

    resultsDF["betaConv"] = betaConv
    resultsDF["betaIter"] = fit["betaIter"]
    resultsDF["deviance"] = -2 * fit["logLike"]
    resultsDF["maxCooks"] = maxCooks

    # if useT need to add the t degrees of freedom to the end of resultsList
    if useT:
        resultsDF["tDegreesFreedom"] = df

    WaldResults = buildDataFrameWithNACols(resultsDF, obj.var["allZero"])
    WaldResults.index = obj.var_names
    assert np.sum(WaldResults.columns.isin(obj.var.columns)) == 0
    new_var = pd.concat([obj.var, WaldResults], axis=1)
    for c in obj.var.columns:
        new_var.type[c] = obj.var.type[c]
        new_var.description[c] = obj.var.description[c]
    obj.var = new_var

    for c in WaldResults.columns:
        obj.var.type[c] = "results"

    obj.var.description["betaConv"] = "convergence of betas"
    obj.var.description["betaIter"] = "iterations for betas"
    obj.var.description["deviance"] = "deviance for the fitted model"
    obj.var.description["maxCooks"] = "maximum Cook's distance for column"

    lfcType = "MAP" if obj.betaPrior else "MLE"
    for c, n in zip(betaMatrix.columns, modelMatrixNames):
        obj.var.description[c] = f"log2 fold change ({lfcType}): {n}"
    for c, n in zip(betaSE.columns, modelMatrixNames):
        obj.var.description[c] = f"standard error: {n}"
    for c, n in zip(WaldStatistic.columns, modelMatrixNames):
        obj.var.description[c] = f"Wald statistic: {n}"

    for c, n in zip(WaldPvalue.columns, modelMatrixNames):
        obj.var.description[c] = f"Wald test p-value: {n}"
    if betaPrior:
        for c, n in zip(mleBetaMatrix.columns, modelMatrixNames):
            obj.var.description[c] = c.replace("_", " ")
    if useT:
        obj.var.description["tDegreesFreedom"] = "t degrees of freedom for Wald test"

    return obj


def calculateCooksDistance(obj, H, modelMatrix):
    p = modelMatrix.shape[1]
    dispersions = robustMethodOfMomentsDisp(obj, modelMatrix)
    V = obj.layers["mu"] + dispersions * obj.layers["mu"] ** 2
    PearsonResSq = (obj.counts() - obj.layers["mu"]) ** 2 / V
    return PearsonResSq / p * H / (1 - H) ** 2


# TODO make it a method of DESeqDataSet
def robustMethodOfMomentsDisp(obj, modelMatrix):
    """calculate a robust method of moments dispersion,
    in order to estimates the dispersion excluding]
    individual outlier counts which would raise the variance estimate.

    Arguments
    ---------
    obj : ??
    modelMatrix : ??
    """
    cnts = obj.counts(normalized=True)
    # if there are 3 or more replactes in any cell
    threeOrMore = nOrMoreInCell(modelMatrix, n=3)
    if np.any(threeOrMore):
        cells = Factor([tuple(modelMatrix[i]) for i in range(modelMatrix.shape[0])])
        levelsThreeOrMore = cells.categories[cells.value_counts() >= 3]
        idx = cells.isin(levelsThreeOrMore)
        cntsSub = cnts[idx, :]
        cellsSub = Factor(cells[idx])
        v = trimmedCellVariance(cntsSub, cellsSub)
    else:
        v = trimmedVariance(cnts)

    m = np.mean(cnts, axis=0)
    alpha = (v - m) / m**2
    # cannot use the typical minDisp = 1e-8 here or else all counts in the same
    # group as the outlier count will get an extrem Cook's distance
    minDisp = 0.04
    alpha = np.maximum(alpha, minDisp)
    return alpha


def trimmedCellVariance(cnts, cells):
    # how much to trim at different n
    trimratio = [1 / 3, 1 / 4, 1 / 8]

    # returns an index for the vector above for three sample size bins
    def trimfn(n):
        if n <= 0:
            return np.nan
        elif n <= 3.5:
            return 0
        elif n <= 23.5:
            return 1
        else:
            return 2

    cellMeans = np.vstack(
        [
            trim_mean(cnts[cells == lvl, :], trimratio[trimfn(n)], axis=0)
            for lvl, n in cells.value_counts().items()
        ]
    )
    assert cellMeans.shape[1] == cnts.shape[1]
    assert cellMeans.shape[0] == len(cells.value_counts())

    qmat = cellMeans[cells.codes, :]
    sqerror = (cnts - qmat) ** 2
    varEst = np.vstack(
        [
            [2.04, 1.86, 1.51][trimfn(n)]
            * trim_mean(sqerror[cells == lvl, :], trimratio[trimfn(n)], axis=0)
            for lvl, n in cells.value_counts().items()
        ]
    )
    assert varEst.shape[1] == sqerror.shape[1]
    assert varEst.shape[0] == len(cells.value_counts())

    # take the max of variance estimates from cells
    # as one condition might have highly variable counts
    return np.max(varEst, axis=0)


def trimmedVariance(x):
    rm = trim_mean(x, 1 / 8, axis=0)
    sqerror = (x - rm) ** 2
    # scale due to trimming of large squares
    return 1.51 * trim_mean(sqerror, 1 / 8, axis=0)


def recordMaxCooks(design, clinicalData, modelMatrix, cooks, numCol):
    """this function breaks out the logic for calculating the max Cook's distance:
    the samples over which max Cook's distance is calculated:

    Cook's distance is considered for those samples with 3 or more replicates per cell

    if m == p or there are no samples over which to calculate max Cook's, return NA
    """
    samplesForCooks = nOrMoreInCell(modelMatrix, n=3)
    m, p = modelMatrix.shape
    if m > p and np.any(samplesForCooks):
        return np.max(cooks[samplesForCooks, :], axis=0)
    else:
        return np.repeat(np.nan, numCol)
