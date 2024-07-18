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


import numpy as np
import scipy.stats
from scipy.stats import trim_mean

from ..utils import LOGGER
from .dispersions import estimateDispersionsGeneEst, estimateDispersionsMAP
from .lrt import nbinomLRT
from .misc import nOrMoreInCell
from .wald import nbinomWaldTest, recordMaxCooks


def refitWithoutOutliers(
    obj,
    test,
    betaPrior,
    full,
    reduced,
    quiet,
    minReplicatesForReplace,
    modelMatrix,
    modelMatrixType,
):
    """ """
    cooks = obj.layers["cooks"]
    obj = replaceOutliers(obj, minReplicates=minReplicatesForReplace)

    # refit without outliers, if there were any replacements
    nrefit = np.nansum(obj.var["replace"])
    if nrefit > 0:
        obj = obj.getBaseMeansAndVariances()
        newAllZero = obj.var["replace"] & obj.var["allZero"]
    # only refit if some of the replacements do not result in all zero counts
    # otherwise, these cases are handled by results()
    if nrefit > 0 and nrefit > np.sum(newAllZero):
        LOGGER.info(
            f"""
        -- replacing outliers and refitting for {nrefit} genes
        -- DESeq argument 'minReplicatesForReplace' = {minReplicatesForReplace}
        -- original counts are preserved in dds.counts()
        """
        )

        # refit on those rows which had replacement
        refitReplace = obj.var["replace"] & ~obj.var["allZero"]
        objSub = obj[:, refitReplace]
        intermediateOrResults = objSub.var.type.filter(
            ["intermediate", "results"]
        ).columns
        objSub.var = objSub.var.drop(intermediateOrResults, axis=1)
        for c in intermediateOrResults:
            del objSub.var.type[c]
            del objSub.var.description[c]

        # estimate gene-wise dispersion
        LOGGER.info("estimating dispersions")
        objSub = estimateDispersionsGeneEst(
            objSub, quiet=quiet, modelMatrix=modelMatrix
        )

        # need to redo fitted dispersion due to changes in base mean
        objSub.var["dispFit"] = obj.dispersionFunction(objSub.var["baseMean"])
        objSub.var.type["dispFit"] = "intermediate"
        objSub.var.description["dispFit"] = "fitted values of dispersion"
        dispPriorVar = obj.dispersionFunction.dispPriorVar

        # estimate dispersion MAP
        objSub = estimateDispersionsMAP(
            objSub, quiet=quiet, dispPriorVar=dispPriorVar, modelMatrix=modelMatrix
        )

        # fit GLM
        LOGGER.info("fitting model and testing")
        if test == "Wald":
            betaPriorVar = obj.betaPriorVar
            objSub = nbinomWaldTest(
                objSub,
                betaPrior=betaPrior,
                betaPriorVar=betaPriorVar,
                quiet=quiet,
                modelMatrix=modelMatrix,
                modelMatrixType=modelMatrixType,
            )
        elif test == "LRT":
            objSub = nbinomLRT(objSub, full=full, reduced=reduced, quiet=quiet)

        obj.var.loc[refitReplace, objSub.var.columns] = objSub.var
        obj.var.loc[newAllZero, obj.var.type.filter("results").columns] = np.nan

        # continue to flag if some conditions have less than minReplicatesForReplace
        if np.all(obj.obs["replaceable"]):
            obj.var["maxCooks"] = np.nan
        else:
            replaceCooks = obj.layers["cooks"].copy()
            replaceCooks[obj.obs["replaceable"]] = 0
            obj.var["maxCooks"] = recordMaxCooks(
                obj.design, obj.obs, obj.dispModelMatrix, replaceCooks, obj.n_vars
            )

    if nrefit > 0:
        # save the counts used for fitting as replaceCounts
        obj.layers["replaceCounts"] = obj.counts().copy()
        obj.layers["replaceCooks"] = obj.layers["cooks"].copy()

        # preserve original counts and Cook's distances
        obj.X = obj.layers["originalCounts"]
        obj.layers["cooks"] = cooks

        # no longer needed
        del obj.layers["originalCounts"]

    return obj


def replaceOutliers(
    obj, trim=0.2, cooksCutoff=None, minReplicates=7, whichSamples=None
):
    """
    Replace outliers with trimmed mean

    Note that this function is called within :func:`DESeq`, so is not necessary
    to call on top of a :func:`DESeq` call. See the documentation for
    :code:`minReplicatesForReplace` in :func:`DESeq`.

    This function replaces outlier counts flagged by extreme Cook's distances,
    as calculated by :func:`DESeq`, :func:`nbinomWaldTest` or
    :func:`nbinomLRT`, with values predicted by the trimmed mean over all
    samples (and adjusted by size factor or normalization factor).  This
    function replaces the counts in the matrix returned by :code:`dds.counts()`
    and the Cook's distances in :code:`dds.layers["cook"]`. Original counts are
    preserved in :code:`dds.layers["originalCounts"]`.

    The :func:`DESeq` function calculates a diagnostic measure called Cook's
    distance for every gene and every sample. The :meth:`.DESeqDataSet.results`
    function then sets the p-values to NA for genes which contain an outlying
    count as defined by a Cook's distance above a threshold. With may degrees
    of freedom, i.e. many more samples than number of parameters to be
    estimated, it might be undesirable to remove entire genes fomr the analysis
    just because their data include a single count outlier.  An alternative
    strategy is to replace the outlier counts with the trimmed mean over all
    samples, adjusted by the size factor or normalization factor for that
    sample. The following simple function performs this replacement for the
    user, for samples which have at least :code:`minReplicates` number of
    replicates (including that sample). For more information on Cook's
    distance, please see the two sections of the module documentation: "Dealing
    with count outliers" and "Count outlier detection".

    Arguments
    ---------
    obj : DESeqDataSet
        a DESeqDataSet that has already been processed by either :func:`DESeq`,
        :func:`nbinomWaldTest` or :func:`nbinomLRT`, and therefore contains a
        matrix of Cook's distances (used to define the outlier counts) in
        :code:`obj.layers["cooks"]`.
    trim : float
        the fraction (0 to 0.5) of observations to be trimmed from each end of
        the normalized counts for a gene before the mean is computed.
    cooksCutoff : float
        the threshold for defining an outlier to be replaced. Defaults to the
        .99 quantile of the :math:`F(p, m-p)` distribution, where :math:`p` is
        the number of parameters and :math:`m` is the number of samples.
    minReplicates : int
        the minimum number of replicate samples necessary to consider a sample
        eligible for replacement (including itself). Outlier counts will not be
        replaced if the sample is in a cell which has less than
        :code:`minReplicates` replicates.
    whichSamples : array-like, optional
        a numeric or logical index to specify which samples should have
        outliers replaced. If missing, this is determined using
        :code:`minReplicates`.

    Return
    ------
    DESeqDataSet
        the input :code:`obj` with replaced counts in the slot returned by
        :meth:`.DESeqDataSet.counts`, and the original counts preserved in
        :code:`obj.layers["originalCounts"]`.
    """
    if obj.modelMatrix is None or "cooks" not in obj.layers:
        raise ValueError(
            "first run DESeq, nbinomWaldTest or nbinomLRT to identify outliers"
        )
    if not isinstance(minReplicates, int):
        raise ValueError(f"invalid value for minReplicates: {minReplicates}")
    if minReplicates < 3:
        raise ValueError(
            "at least 3 replicates are necessary in order to identify a sample as a count outlier."
        )

    p = obj.modelMatrix.shape[1]
    m = obj.n_obs
    if m <= p:
        obj.layers["originalCounts"] = obj.counts().copy()
        return obj
    if cooksCutoff is None:
        cooksCutoff = scipy.stats.f.ppf(0.99, p, m - p)

    idx = obj.layers["cooks"] > cooksCutoff
    obj.var["replace"] = np.any(obj.layers["cooks"] > cooksCutoff, axis=0)
    obj.var.type["replace"] = "intermediate"
    obj.var.description["replace"] = "had counts replaced"

    trimBaseMean = trim_mean(obj.counts(normalized=True), trim)
    # build a matrix of counts based on the trimmed mean and the size factors
    if obj.normalizationFactors is not None:
        replacementCounts = (
            np.repeat(trimBaseMean[None], obj.n_obs, axis=0)
            * obj.normalizationFactors.values[:, None]
        )
    else:
        replacementCounts = (
            np.repeat(trimBaseMean[None], obj.n_obs, axis=0)
            * obj.sizeFactors.values[:, None]
        )

    # replace only those values which fall above the cutoff on Cook's distance
    newCounts = obj.counts().copy()
    newCounts[idx] = replacementCounts[idx]

    if whichSamples is None:
        whichSamples = nOrMoreInCell(obj.modelMatrix, n=minReplicates)

    whichSamples.index = obj.obs_names
    obj.obs["replaceable"] = whichSamples
    obj.obs.type["replaceable"] = "intermediate"
    obj.obs.description["replaceable"] = "outliers can be replaced"
    obj.layers["originalCounts"] = obj.counts().copy()
    obj.X[whichSamples] = newCounts[whichSamples]
    return obj
