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

# This file is based on the file 'R/results.R' of the Bioconductor DESeq2
# package (version 3.16).


import numpy as np
import pandas as pd
import patsy
import scipy.stats
from statsmodels.stats.multitest import multipletests
from statsmodels.nonparametric.smoothers_lowess import lowess

from ..utils import pnorm, pt
from .. import __version__
from .deseq2_cpp import fitBeta
from .misc import buildDataFrameWithNACols


def p_adjust(*args, **kwargs):
    return multipletests(*args, **kwargs)[1]


class DESeqResults(pd.DataFrame):
    _metadata = [
        "priorInfo",
        "filterThreshold",
        "filterTheta",
        "filterNumRej",
        "lo_fit",
        "alpha",
    ]

    @property
    def _constructor(self):
        return DESeqResults

    @property
    def _constructor_sliced(self):
        return pd.Series

    def __init__(self, df, priorInfo=None, *args, **kwargs):
        super().__init__(df, *args, **kwargs)
        if priorInfo is not None:
            self.priorInfo = priorInfo


def results_dds(
    obj,
    contrast=None,
    name=None,
    lfcThreshold=0,
    altHypothesis="greaterAbs",
    listValues=(1, -1),
    cooksCutoff=None,
    independentFiltering=True,
    alpha=0.1,
    filter=None,
    theta=None,
    pAdjustMethod="fdr_bh",
    filterRun=None,
    saveCols=None,
    test=None,
    addMLE=False,
    tidy=False,
    parallel=False,
    minmu=0.5,
):
    """
    TODO
    """
    if altHypothesis not in ["greaterAbs", "lessAbs", "greater", "less"]:
        raise ValueError(f"invalid value for altHypothesis: {altHypothesis}")
    if test not in [None, "Wald", "LRT"]:
        raise ValueError(f"invalid value for test: {test}")

    # initial argument testing
    if lfcThreshold < 0:
        raise ValueError("lfcThreshold should be positive")
    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha should be between 0 and 1")
    if not isinstance(listValues, (tuple, list)) or len(listValues) != 2:
        raise ValueError("listValues should be of length 2")
    if listValues[0] <= 0 or listValues[1] >= 0:
        raise ValueError("listValues should contain a positive and a negative number")
    if obj.var.type.filter(["results"]).empty:
        raise ValueError("could not find results in obj. first run DESeq()")
    if test is None:
        test = obj.test
    elif test == "Wald" and obj.test == "LRT":
        # initially test was LRT, now need to add Wald statistics and p-values
        obj = makeWaldTest(obj)
    elif test == "LRT" and obj.test == "Wald":
        raise ValueError(
            "the LRT requires the user to run nbinomLRT or DESeq(obj, test='LRT')"
        )
    if lfcThreshold == 0 and altHypothesis == "lessAbs":
        raise ValueError(
            "when testing altHypothesis='lessAbs', set the argument lfcThreshold to a positive value"
        )

    if addMLE:
        if obj.betaPrior is None:
            raise ValueError(
                "addMLE=True is only for when a beta prior was used. Otherwise, the log2 fold changes are already MLE"
            )
        if name is not None and contrast is None:
            raise ValueError(
                "addMLE=True should be used by providing string vector of length 3 to 'contrast' instead of using 'name'"
            )

    if saveCols is not None:
        try:
            obj.obs[saveCols]
        except:
            raise ValueError(f"invalid value for saveCols: {saveCols}")

    hasIntercept = patsy.INTERCEPT in obj.design.design_info.terms
    isExpanded = obj.modelMatrixType == "expanded"
    noInteraction = all(len(t.factors) < 2 for t in obj.design.design_info.terms)
    # if no intercept was used or an expanded model matrix was used,
    # and neither 'contrast' nor 'name' were specified,
    # and no interactions...
    # then we create the result table: last / first level for last variable
    if (
        test == "Wald"
        and (isExpanded or not hasIntercept)
        and contrast is None
        and name is None
        and noInteraction
    ):
        designVars = [v for v in obj.design.design_info.factor_infos.values()]
        lastVarName = designVars[-1].factor.name()
        lastVar = obj.obs[lastVarName]
        if designVars[-1].type == "categorical":
            contrast = [
                lastVarName,
                f"{lastVar.dtype.categories[-1]}",
                f"{lastVar.dtype.categories[0]}",
            ]

    if name is None:
        name = lastCoefName(obj)
    else:
        if not isinstance(name, str):
            raise ValueError("'name' should be a string")

    # done with input argument testing

    WaldResults = f"WaldPvalue_{name}" in obj.obs
    LRTResults = "LRTPValue" in obj.obs

    # this will be used in cleanContrast, and in the lfcThreshold chunks below
    useT = "tDegreesFreedom" in obj.var

    # if performing a contrast call the function cleanContrast()
    if contrast is not None:
        resNames = obj.resultsNames()
        # do some arg checking/cleaning
        contrast = checkContrast(contrast, resNames)

        ### cleanContrast call ###
        # need to go back to C++ code in order to build the beta covariance matrix
        # then this is multiplied by the numeric contrast to get the Wald statistic.
        # with 100s of samples, this can get slow, so offer parallelization
        if not parallel:
            res = cleanContrast(
                obj,
                contrast,
                expanded=isExpanded,
                listValues=listValues,
                test=test,
                useT=useT,
                minmu=minmu,
            )
        else:
            raise NotImplementedError("parallelization not implemented yet")

    else:
        # if not performing a contrast
        # pull relevant columns from obj.var
        log2FoldChange = getCoef(obj, name)
        lfcSE = getCoefSE(obj, name)
        stat = getStat(obj, name, test)
        pvalue = getPvalue(obj, name, test)
        res = pd.DataFrame(
            {
                "baseMean": obj.var["baseMean"],
                "log2FoldChange": log2FoldChange,
                "lfcSE": lfcSE,
                "stat": stat,
                "pvalue": pvalue,
            }
        )

    res.index = obj.var_names

    # add unshrunken MLE coefficients to the results table
    if addMLE:
        if not isinstance(contrast, (tuple, list)) or len(contrast) != 3:
            raise ValueError(
                "addMLE only implemented for contrast=['condition', 'B', 'A']"
            )
        res = pd.concat([res, mleContrast(obj, contrast)], axis=1)
        res = res[["baseMean", "log2FoldChange", "lfcMLE", "lfcSE", "stat", "pvalue"]]
        # if an all zero contrast, also zero out the lfcMLE
        res["lfcMLE"][(res["log2FoldChange"] == 0) & (res["stat"] == 0)] = 0

    # only if we need to generate new p-values
    if not (lfcThreshold == 0 and altHypothesis == "greaterAbs"):
        if test == "LRT":
            raise ValueError(
                "tests of log fold change above or below a threshold must be Wald test."
            )
        # check requirement if betaPrior was set to False
        if altHypothesis == "lessAbs" and obj.betaPrior:
            raise ValueError(
                "testing altHypothesis='lessAbs' requires setting the DESeq() argument betaPrior=False"
            )

        # easier to read
        LFC = res.log2FoldChange
        SE = res.lfcSE
        T = lfcThreshold

        if useT:
            df = obj.var["tDegreesFreedom"]
            pfunc = lambda q: pt(q, df=df, lower_tail=False)
        else:
            pfunc = lambda q: pnorm(q, lower_tail=False)

        if altHypothesis == "greaterAbs":
            newStat = np.sign(LFC) * np.maximum((np.abs(LFC) - T) / SE, 0)
            newPvalue = np.minimum(1, 2 * pfunc((np.abs(LFC) - T) / SE))
        elif altHypothesis == "lessAbs":
            newStatAbove = np.maximum((T - LFC) / SE, 0)
            pvalueAbove = pfunc((T - LFC) / SE)
            newStatBelow = np.maximum((LFC + T) / SE, 0)
            pvalueBelow = pfunc((LFC + T) / SE)
            newStat = np.minimum(newStatAbove, newStatBelow)
            newPvalue = np.maximum(pvalueAbove, pvalueBelow)
        elif altHypothesis == "greater":
            newStat = np.maximum((LFC - T) / SE, 0)
            newPvalue = pfunc((LFC - T) / SE)
        elif altHypothesis == "less":
            newStat = np.minimum((LFC + T) / SE, 0)
            newPvalue = pfunc((-T - LFC) / SE)

        res.stat = newStat
        res.pvalue = newPvalue

    # calculate Cook's cutoff
    m, p = obj.dispModelMatrix.shape

    if cooksCutoff is None or (isinstance(cooksCutoff, bool) and cooksCutoff):
        cooksCutoff = scipy.stats.f.ppf(0.99, p, m - p)

    # apply cutoff based on maximum Cook's distance
    # NB: cooksCutoff is not necessarily a Boolean
    performCooksCutoff = not (cooksCutoff == False)
    if performCooksCutoff:
        cooksOutlier = obj.var["maxCooks"] > cooksCutoff

        ### BEGIN heuristic to avoid filtering genes with low count outliers
        # as according to Cook's cutoff. only for two group designs.
        # do not filter if three or more counts are larger
        if np.any(cooksOutlier[~np.isnan(cooksOutlier)]):
            designVars = obj.design.design_info.factor_infos
            if len(designVars) == 1:
                var = [v for v in designVars.values()][0]
                if var.type == "categorical" and len(var.categories) == 2:
                    maxIndices = np.argmax(obj.layers["cooks"][:, cooksOutlier], axis=0)
                    outliers = obj.counts()[:, cooksOutlier]
                    # counts for the outliers with max cooks
                    outCount = np.take_along_axis(
                        outliers, maxIndices[None], 0
                    ).squeeze()
                    # if three or more counts larger than the outlier
                    # do not filter out the p-value for those genes
                    dontFilter = np.sum(outliers > outCount, axis=0) >= 3
                    # reset the outlier status for these genes
                    cooksOutlier[cooksOutlier] &= ~dontFilter
        ### END heuristic

        res.pvalue[cooksOutlier] = np.nan

    # if original baseMean was positive, but now zero due to replaced counts,
    # fill in results
    if "replace" in obj.var and np.nansum(obj.var["replace"]) > 0:
        nowZero = obj.var["replace"] & (obj.var["baseMean"] == 0)
        res.log2FoldChange[nowZero] = 0
        if addMLE:
            res.lfcMLE[nowZero] = 0
        res.lfcSE[nowZero] = 0
        res.stat[nowZero] = 0
        res.pvalue[nowZero] = 1

    # add prior information
    if not obj.betaPrior:
        priorInfo = {
            "type": "none",
            "package": "inmoose.deseq2",
            "version": __version__,
        }
    else:
        priorInfo = {
            "type": "normal",
            "package": "inmoose.deseq2",
            "version": __version__,
            "betaPriorVar": obj.betaPriorVar,
        }

    # make results object
    deseqRes = DESeqResults(res, priorInfo=priorInfo)

    # p-value adjustment
    if filterRun is None:
        deseqRes = pvalueAdjustment(
            deseqRes, independentFiltering, filter, theta, alpha, pAdjustMethod
        )
    else:
        deseqRes = filterRun(deseqRes, filter, alpha, pAdjustMethod)

    # stash lfcThreshold
    deseqRes.lfcThreshold = lfcThreshold

    # remove rownames and attach a new column 'row'
    if tidy:
        raise NotImplementedError()

    if saveCols is not None:
        mrows2Save = obj.obs[saveRows, :]
        deseqRes.mrows2Save = mrows2Save

    return deseqRes


def lastCoefName(obj):
    """convenience function to guess the name of the last coefficient
    in the model matrix, unless specified this will be used for plots and
    accessor functions.
    """
    return obj.resultsNames()[-1]


def getCoef(obj, name):
    if name is None:
        name = lastCoefName(obj)
    return obj.var[name]


def getCoefSE(obj, name):
    if name is None:
        name = lastCoefName(obj)
    return obj.var[f"SE_{name}"]


def getStat(obj, name, test="Wald"):
    if name is None:
        name = lastCoefName(obj)

    if test == "Wald":
        return obj.var[f"WaldStatistic_{name}"]
    elif test == "LRT":
        return obj.var["LRTStatistic"]
    else:
        raise ValueError(f"unknown test: {test}")


def getPvalue(obj, name, test="Wald"):
    if name is None:
        name = lastCoefName(obj)

    if test == "Wald":
        return obj.var[f"WaldPvalue_{name}"]
    elif test == "LRT":
        return obj.var["LRTPvalue"]
    else:
        raise ValueError(f"unknown test: {test}")


def pvalueAdjustment(res, independentFiltering, filter, theta, alpha, pAdjustMethod):
    # perform independent filtering
    if independentFiltering:
        if filter is None:
            filter = res.baseMean
        if theta is None:
            lowerQuantile = np.mean(filter == 0)
            if lowerQuantile < 0.95:
                upperQuantile = 0.95
            else:
                upperQuantile = 1.0
            theta = np.linspace(lowerQuantile, upperQuantile, 50)

        # do filtering using genefilter
        if len(theta) <= 1:
            raise ValueError("theta should be a list")
        if len(filter) != res.shape[0]:
            raise ValueError("filter should have as many elements as res has rows")
        filtPadj = filtered_p(
            filter_=filter, test=res.pvalue, theta=theta, method=pAdjustMethod
        )
        numRej = np.nansum(filtPadj < alpha, axis=0)
        # prevent over-aggressive filtering when all genes are null,
        # by requiring the max number of rejections is above a fitted curve.
        # If the max number of rejection is not greater than 10, then don't
        # perform independent filtering at all.
        lo_fit = lowess(numRej, theta, frac=1 / 5)
        if np.max(numRej) <= 10:
            j = 0
        else:
            if np.all(numRej == 0):
                residual = 0
            else:
                residual = numRej[numRej > 0] - lo_fit[numRej > 0, 1]
            thresh = np.max(lo_fit[:, 1]) - np.sqrt(np.mean(residual**2))
            if np.any(numRej > thresh):
                j = np.nonzero(numRej > thresh)[0][0]
            else:
                j = 0

        padj = filtPadj[:, j]
        cutoffs = np.quantile(filter, theta)
        filterThreshold = cutoffs[j]
        filterNumRej = pd.DataFrame({"theta": theta, "numRej": numRej})
        filterTheta = theta[j]

        res.filterThreshold = filterThreshold
        res.filterTheta = filterTheta
        res.filterNumRej = filterNumRej
        res.lo_fit = lo_fit
        res.alpha = alpha

    else:
        # regular p-value adjustment
        # does not include those rows which were removed
        # by maximum Cook's distance
        padj = p_adjust(res.pvalue, method=pAdjustMethod)

    res["padj"] = padj
    res.type["padj"] = "results"
    res.description["padj"] = f"{pAdjustMethod} adjusted p-values"

    return res


def filtered_p(filter_, test, theta, method, data=None):
    if callable(filter_):
        U1 = filter_(data)
    else:
        U1 = filter_

    cutoffs = np.quantile(U1, theta)
    result = np.full((len(U1), len(cutoffs)), np.nan)
    for i in range(len(cutoffs)):
        use = U1 >= cutoffs[i]
        if np.any(use):
            if callable(test):
                U2 = test(data[use, :])
            else:
                U2 = test[use]
            result[use, i] = p_adjust(U2, method=method)
    return result


def getContrast(obj, contrast, useT, minmu):
    """takes a DESeqDataSet obj and a numeric vector specifying a contrast
    and returns a vector of Wald statistics corresponding to the contrast.
    """
    if contrast is None:
        raise ValueError("must provide a contrast")
    modelMatrix = obj.modelMatrix

    # only continue on the cols with non-zero col mean
    objNZ = obj[:, ~obj.var["allZero"]]
    normalizationFactors = objNZ.getSizeOrNormFactors()
    alpha_hat = objNZ.var["dispersion"]
    # convert beta to log scale
    beta_mat = np.log(2) * objNZ.var.description.filter(regex="log2 fold change")
    # convert beta prior variance to log scale
    lambda_ = 1 / (np.log(2) ** 2 * obj.betaPriorVar)

    # check if DESeq() replaced outliers
    if "replaceCounts" in obj.layers:
        countsMatrix = objNZ.layers["replaceCounts"]
    else:
        countsMatrix = objNZ.counts()

    # use weights if they are present
    if "weights" in obj.layers:
        useWeights = True
        weights = obj.layers["weights"]
        if not np.all(weights >= 0):
            raise ValueError("all weights must be positive")
        weights = weights / np.max(weights, axis=0)
    else:
        useWeights = False
        weights = np.ones(obj.shape)

    betaRes = fitBeta(
        y=countsMatrix,
        x=modelMatrix,
        nf=normalizationFactors,
        alpha_hat=alpha_hat,
        contrast=contrast,
        beta_mat=beta_mat.values,
        lambda_=lambda_,
        weights=weights,
        useWeights=useWeights,
        tol=1e-8,
        maxit=0,
        useQR=False,  # QR not relevant, fitting loop is not entered
        minmu=minmu,
    )
    # convert back to log2 scale
    contrastEstimate = np.log2(np.exp(1)) * betaRes["contrast_num"]
    contrastSE = np.log2(np.exp(1)) * betaRes["contrast_denom"]
    contrastStatistic = contrastEstimate / contrastSE

    if useT:
        if "tDegreesFreedom" not in obj.var:
            raise ValueError("tDegreesFreedom should be in obj.var")
        df = objNZ.var["tDegreesFreedom"]
        contrastPvalue = 2 * pt(np.abs(contrastStatistic), df=df, lower_tail=False)
    else:
        contrastPvalue = 2 * pnorm(np.abs(contrastStatistic), lower_tail=False)

    contrastResults = pd.DataFrame(
        {
            "log2FoldChange": contrastEstimate,
            "lfcSE": contrastSE,
            "stat": contrastStatistic,
            "pvalue": contrastPvalue,
        }
    )
    contrastResults = buildDataFrameWithNACols(contrastResults, obj.var["allZero"])
    contrastResults.index = obj.var_names
    return contrastResults


def cleanContrast(obj, contrast, expanded, listValues, test, useT, minmu):
    """this function takes a desired contrast as specified by results(),
    performs checks, and then either returns the already existing contrast
    or generates the contrast by calling getContrast() using a numeric vector
    """
    # get the names of columns in the beta matrix
    resNames = obj.resultsNames()
    # if possible, return pre-computed columns, which are
    # already stored in obj.var. This will be the case using
    # results() with 'name', or if expanded model matrices were not
    # run and the contrast contains the reference level as numerator and denominator

    resReady = False

    if all(isinstance(c, str) for c in contrast):
        contrastFactor = contrast[0]
        if contrastFactor not in obj.obs:
            raise ValueError(
                f"{contrastFactor} should be the name of a factor in the obs data of the DESeqDataSet"
            )
        if not isinstance(obj.obs[contrastFactor].dtype, pd.CategoricalDtype):
            raise ValueError(f"{contrastFactor} is not a factor")

        contrastNumLevel = contrast[1]
        contrastDenomLevel = contrast[2]
        # make sure contrastBaseLevel is a string
        contrastBaseLevel = f"{obj.obs[contrastFactor].dtype.categories[0]}"

        # check for intercept
        hasIntercept = 0 in [len(t.factors) for t in obj.design.design_info.terms]
        firstVar = (
            contrastFactor
            == [t.name() for t in obj.design.design_info.terms if len(t.factors) > 0][0]
        )

        # tricky case: if the design has no intercept, the factor is not the
        # first variable in the design, and one of the numerator or
        # denominator is the reference level, then the desired contrast is
        # simply a coefficient (or -1 times)
        noInterceptPullCoef = (
            not hasIntercept
            and not firstVar
            and (contrastBaseLevel in [contrastNumLevel, contrastDenomLevel])
        )

        # case 1: standard model matrices: pull coef or build the appropriate contrast
        # coefficients names are of the form "factor_level_vs_baselevel"
        # output: contrastNumColumn and contrastDenomColumn
        if not expanded and (hasIntercept or noInterceptPullCoef):
            contrastNumColumn = (
                f"{contrastFactor}_{contrastNumLevel}_vs_{contrastBaseLevel}"
            )
            contrastDenomColumn = (
                f"{contrastFactor}_{contrastDenomLevel}_vs_{contrastBaseLevel}"
            )
            # check that the desired contrast is already available in obj.var,
            # and then we can either take it directly or multiply the log fold
            # change and Wald stat by -1
            if contrastDenomLevel == contrastBaseLevel:
                cleanName = (
                    f"{contrastFactor} {contrastNumLevel} vs {contrastDenomLevel}"
                )
                # the results can be pulled directly from obj.var
                if hasIntercept or not firstVar:
                    name = (
                        f"{contrastFactor}_{contrastNumLevel}_vs_{contrastDenomLevel}"
                    )
                else:
                    name = f"{contrastFactor}[{contrastNumLevel}]"

                if name not in resNames:
                    raise ValueError(
                        f"as {contrastDenomLevel} is the reference level, was expecting {name} to be present in 'resultsNames(obj)'"
                    )

                log2FoldChange = getCoef(obj, name)
                lfcSE = getCoefSE(obj, name)
                stat = getStat(obj, name, test)
                pvalue = getPvalue(obj, name, test)
                res = pd.DataFrame(
                    {
                        "baseMean": obj.var["baseMean"],
                        "log2FoldChange": log2FoldChange,
                        "lfcSE": lfcSE,
                        "stat": stat,
                        "pvalue": pvalue,
                    }
                )
                lfcType = "MAP" if obj.betaPrior else "MLE"
                res.description[
                    "log2FoldChange"
                ] = f"log2 fold change ({lfcType}): {cleanName}"
                resReady = True

            elif contrastNumLevel == contrastBaseLevel:
                # fetch the results for denom vs num
                # and multiply the log fold change and stat by -1
                cleanName = (
                    f"{contrastFactor} {contrastNumLevel} vs {contrastDenomLevel}"
                )
                if hasIntercept or not firstVar:
                    swapName = (
                        f"{contrastFactor}_{contrastDenomLevel}_vs_{contrastNumLevel}"
                    )
                else:
                    swapName = f"{contrastFactor}[{contrastDenomLevel}]"

                if swapName not in resNames:
                    raise ValueError(
                        f"as {contrastNumLevel} is the reference level, was expecting {swapName} to be present in resultsNames(obj)"
                    )

                log2FoldChange = getCoef(obj, swapName)
                lfcSE = getCoefSE(obj, swapName)
                stat = getStat(obj, swapName, test)
                pvalue = getPvalue(obj, swapName, test)
                res = pd.DataFrame(
                    {
                        "baseMean": obj.var["baseMean"],
                        "log2FoldChange": log2FoldChange,
                        "lfcSE": lfcSE,
                        "stat": stat,
                        "pvalue": pvalue,
                    }
                )
                res["log2FoldChange"] *= -1
                if test == "Wald":
                    res["stat"] *= -1
                lfcType = "MAP" if obj.betaPrior else "MLE"
                res.description[
                    "log2FoldChange"
                ] = f"log2 fold change ({lfcType}): {cleanName}"
                res.description["lfcSE"] = f"standard error: {cleanName}"
                # rename some of the columns using the flipped contrast
                if test == "Wald":
                    res.description["stat"] = f"Wald statistic: {cleanName}"
                    res.description["pvalue"] = f"Wald test p-value: {cleanName}"

                resReady = True

            else:
                # check for the case where neither are present
                # as comparisons against reference level
                if not (
                    contrastNumColumn in resNames and contrastDenomColumn in resNames
                ):
                    raise ValueError(
                        f"{contrastNumLevel} and {contrastDenomLevel} should be levels of {contrastFactor} such that {contrastNumLevel} and {contrastDenomLevel} are contained in 'resultsNames(obj)'"
                    )

        # case 2: expanded model matrices or no intercept and first variable
        # need to then build the appropriate contrast.
        # these coefficient names have the form "factorLevel"
        # output: contrastNumColumn and contrastDenomColumn
        else:
            # we only need to check validity
            if hasIntercept:
                contrastNumColumn = f"{contrastFactor}[T.{contrastNumLevel}]"
                contrastDenomColumn = f"{contrastFactor}[T.{contrastDenomLevel}]"
            elif not firstVar:
                contrastNumColumn = (
                    f"{contrastFactor}_{contrastNumLevel}_vs_{contrastBaseLevel}"
                )
                contrastDenomColumn = (
                    f"{contrastFactor}_{contrastDenomLevel}_vs_{contrastBaseLevel}"
                )
            else:
                contrastNumColumn = f"{contrastFactor}[{contrastNumLevel}]"
                contrastDenomColumn = f"{contrastFactor}[{contrastDenomLevel}]"
            if not (contrastNumColumn in resNames and contrastDenomColumn in resNames):
                raise ValueError(
                    f"{contrastNumColumn} and {contrastDenomColumn} are expected to be in resultsNames(obj): {resNames}"
                )

        # check if both level have all zero counts
        # (this has to be done here to make use of error checking above)
        contrastAllZero = contrastAllZeroCharacter(
            obj, contrastFactor, contrastNumLevel, contrastDenomLevel
        )

    # if the result table not already built in the above code
    if not resReady:
        # here, a numeric / list / string contrast which will be converted
        # into a numeric contras and run through getContrast()
        if all(isinstance(c, (int, float, np.number)) for c in contrast):
            # make name for numeric contrast
            signMap = ["", "", "+"]
            contrastSigns = [signMap[x + 1] for x in np.sign(contrast)]
            contrastName = ",".join(
                [f"{s}{c}" for s, c in zip(contrastSigns, contrast)]
            )
            # make sure the contrast is an np array
            contrast = np.asarray(contrast)
        elif all(isinstance(c, str) for c in contrast):
            # interpret string contrast into numeric and make a name for the contrast
            contrastNumeric = np.zeros(len(resNames))
            contrastNumeric[resNames == contrastNumColumn] = 1
            contrastNumeric[resNames == contrastDenomColumn] = -1
            contrast = contrastNumeric
            contrastName = (
                f"{contrastFactor} {contrastNumLevel} vs {contrastDenomLevel}"
            )
        else:
            # interpret 2-list contrast into numeric and make a name for the contrast
            lc1 = len(contrast[0])
            lc2 = len(contrast[1])
            # these just used for naming
            listvalname1 = round(listValues[0], 3)
            listvalname2 = round(listValues[1], 3)
            if lc1 > 0 and lc2 > 0:
                listvalname2 = np.abs(listvalname2)
                listvalname1 = "" if listvalname1 == 1 else f"{listvalname1} "
                listvalname2 = "" if listvalname2 == 1 else f"{listvalname2} "
                contrastName = f"{listvalname1}{'+'.join(contrast[0])} vs {listvalname2}{'+'.join(contrast[1])}"
            elif lc1 > 0 and lc2 == 0:
                listvalname1 = "" if listvalname1 == 1 else f"{listvalname1} "
                contrastName = f"{listvalname1}{'+'.join(contrast[0])} effect"
            elif lc1 == 0 and lc2 > 0:
                contrastName = f"{listvalname2}{'+'.join(contrast[1])} effect"

            contrastNumeric = np.zeros(len(resNames))
            contrastNumeric[resNames.isin(contrast[0])] = listValues[0]
            contrastNumeric[resNames.isin(contrast[1])] = listValues[1]
            contrast = contrastNumeric

        contrastAllZero = contrastAllZeroNumeric(obj, contrast)

        # now get the contrast
        res = getContrast(obj, contrast, useT=useT, minmu=minmu)
        lfcType = "MAP" if obj.betaPrior else "MLE"
        for c in res.columns:
            res.type[c] = "results"
        res.description[
            "log2FoldChange"
        ] = f"log2 fold change ({lfcType}): {contrastName}"
        res.description["lfcSE"] = f"standard error: {contrastName}"
        res.description["stat"] = f"Wald statistic: {contrastName}"
        res.description["pvalue"] = f"Wald test p-value: {contrastName}"
        res["baseMean"] = obj.var["baseMean"]
        res.type["baseMean"] = obj.var.type["baseMean"]
        res.description["baseMean"] = obj.var.description["baseMean"]

    # if the counts in all samples included in contrast are zero
    # then zero out the LFC, Wald stat and p-value set to 1
    contrastAllZero = contrastAllZero & ~obj.var["allZero"]
    contrastAllZero.index = res.index
    if np.sum(contrastAllZero) > 0:
        res.log2FoldChange[contrastAllZero] = 0
        res.stat[contrastAllZero] = 0
        res.pvalue[contrastAllZero] = 1

    # if test is "LRT", overwrite the statistic and p-value
    # (we only ran contrast for the coefficient)
    if test == "LRT":
        stat = getStat(obj, name=None, test=test)
        pvalue = getPvalue(obj, name=None, test=test)
        res = res[["baseMean", "log2FoldChange", "lfcSE"]]
        res["stat"] = stat
        res["pvalue"] = pvalue

    return res


def mleContrast(obj, contrast):
    contrastFactor = contrast[0]
    contrastNumLevel = contrast[1]
    contrastDenomLevel = contrast[2]
    contrastRefLevel = obj.obs[contrastFactor].dtype.categories[0]
    contrastNumColumn = f"MLE_{contrastFactor}_{contrastNumLevel}_vs_{contrastRefLevel}"
    contrastDenomColumn = (
        f"MLE_{contrastFactor}_{contrastDenomLevel}_vs_{contrastRefLevel}"
    )
    cleanName = f"log2 fold change (MLE): {contrastFactor} {contrastNumLevel} vs {contrastDenomLevel}"

    if contrastDenomLevel == contrastRefLevel:
        name = f"MLE_{contrastFactor}_{contrastNumLevel}_vs_{contrastDenomLevel}"
        lfcMLEName = name
        lfcMLE = obj.var[name]
    elif contrastNumLevel == contrastRefLevel:
        swapName = f"MLE_{contrastFactor}_{contrastDenomLevel}_vs_{contrastNumLevel}"
        lfcMLEName = swapName
        lfcMLE = -1 * obj.var[swapName]
    else:
        numMLE = obj.var[contrastNumColumn]
        denomMLE = obj.var[contrastDenomColumn]
        lfcMLEName = contrastNumColumn
        lfcMLE = numMLE - denomMLE

    res = pd.DataFrame({"lfcMLE": lfcMLE})
    res.type["lfcMLE"] = obj.var.type[lfcMLEName]
    res.description["lfcMLE"] = cleanName
    return res


def is_iterable(x):
    try:
        return not isinstance(x, str) and all(True for y in x)
    except TypeError:
        return False


def checkContrast(contrast, resNames):
    if not is_iterable(contrast):
        contrast = [contrast]
    is_numeric = all(isinstance(c, (int, float, np.number)) for c in contrast)
    is_string = all(isinstance(c, str) for c in contrast)

    if is_string and len(contrast) == 3:
        if contrast[1] == contrast[2]:
            raise ValueError(
                f"{contrast[1]} and {contrast[2]} should be different level names"
            )

    elif is_numeric:
        if len(contrast) != len(resNames):
            raise ValueError(
                "numeric contrast vector should have one element for every element of 'resultsNames(obj)'"
            )
        if np.all(np.equal(contrast, 0)):
            raise ValueError(
                "numeric contrast vector cannot have all elements equal to 0"
            )
    else:
        if len(contrast) == 1:
            contrast = [contrast[0], []]
        if len(contrast) != 2:
            raise ValueError(
                "'contrast', as a pair of lists, should have length 2, or if length 1, an empty list will be added as the second element"
            )

        # make sure contrast is a pair of iterables
        contrast = [c if is_iterable(c) else [c] for c in contrast]

        if not all(isinstance(c, str) for cc in contrast for c in cc):
            raise ValueError(
                "'contrast', as a pair of lists, should have lists of strings as elements"
            )
        if not all(c in resNames for cc in contrast for c in cc):
            raise ValueError(
                "all elements of the 2-element contrast should be elements of 'resultsNames(obj)'"
            )
        if len(np.intersect1d(contrast[0], contrast[1])) > 0:
            raise ValueError(
                "elements in the 2-element contrast should only appear in the numerator (first element of contrast) or the denominator(second element), but not both"
            )
        if len(contrast[0]) + len(contrast[1]) == 0:
            raise ValueError(
                "one the 2 elements in the list should be contain at least one element"
            )

    return contrast


def contrastAllZeroCharacter(obj, contrastFactor, contrastNumLevel, contrastDenomLevel):
    cts = obj.counts()
    f = obj.obs[contrastFactor]
    if np.issubdtype(f.dtype.categories.dtype, np.number):
        contrastNumLevel = int(contrastNumLevel)
        contrastDenomLevel = int(contrastDenomLevel)
    cts_sub = cts[f.isin([contrastNumLevel, contrastDenomLevel])]
    return np.sum(cts_sub == 0, axis=0) == cts_sub.shape[0]


def contrastAllZeroNumeric(obj, contrast):
    modelMatrix = obj.modelMatrix
    # this extra leg-work to zero out FLC, lfcSE, and set p-value to 1
    # for contrasts comparing groups where both groups have all zeros.

    # it is only implemented for the case in which we can identify
    # the relevant samples by multiplying the model matrix
    # with a vector where the non-zero elements of the numeric contrast
    # are replaced with 1.

    # so, this code will not zero out in the case of standard model matrices
    # where the user supplies a numeric contrast that pulls out a single column
    # of the model matrix, for example.

    if np.all(np.greater_equal(contrast, 0)) or np.all(np.less_equal(contrast, 0)):
        return np.repeat(False, obj.n_vars)

    contrastBinary = np.where(contrast == 0, 0, 1)
    whichSamples = np.where(modelMatrix @ contrastBinary == 0, 0, 1)
    zeroTest = obj.counts().T @ whichSamples
    return zeroTest == 0
