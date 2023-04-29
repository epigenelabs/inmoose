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
import pandas as pd
import scipy.stats

from .Hmisc import wtd_quantile
from .misc import renameModelMatrixColumns


def estimateBetaPriorVar(
    obj, betaPriorMethod="weighted", upperQuantile=0.05, modelMatrix=None
):
    """
    steps for estimating the beta prior variance

    This lower-level function is called within :func:`DESeq` or
    :func:`nbinomWaldTest`. End users should use those higher-level functions.

    Arguments
    ---------
    obj : DESeqDataSet
        a :class:`.DESeqDataSet`
    betaPriorMethod : { "weighted", "quantile" }
        the method for calculating the beta prior variance:

        - :code:`"quantile"` matches a normal distribution using the upper
          quantile of the finite MLE betas.
        - :code:`"weighted"` matches a normal distribution using the upper
          quantile, but weighting by the variance of the MLE betas.

    upperQuantile : float
        the upper quantile to be used for the method of beta prior variance
        estimation
    modelMatrix : design matrix
        an optional design matrix, typically left :code:`None` and built within
        the function

    Returns
    -------
    ndarray
        the vector of variances for the prior on the beta in the :func:`DESeq`
        GLM
    """
    objNZ = obj[:, ~obj.var["allZero"]]

    betaMatrix = objNZ.var.filter(regex="MLE_")
    colnamesBM = [s.replace("MLE_", "") for s in betaMatrix.columns]
    # renaming in reverse:
    # make these standard colnames as from patsy.dmatrix
    convertNames = renameModelMatrixColumns(obj.obs, obj.design)
    convertNames = {y: x for x, y in convertNames.items()}
    colnamesBM = pd.Index(
        [convertNames[x] if x in convertNames else x for x in colnamesBM]
    )
    betaMatrix.columns = colnamesBM

    # this is the model matrix from an MLE run
    if modelMatrix is None:
        modelMatrix = obj.design
    modelMatrixType = obj.modelMatrixType

    if betaPriorMethod not in ["weighted", "quantile"]:
        raise ValueError(f"invalid value for betaPriorMethod: {betaPriorMethod}")

    # estimate the variance of the prior on betas
    # if expanded, first calculate LFC for all possible contrasts
    if modelMatrixType == "expanded":
        betaMatrix = objNZ.addAllContrasts(betaMatrix)

    # weighting by 1/Var(log(K))
    # Var(log(K)) ~ Var(K)/mu**2 = 1/mu + alpha
    # and using the fitted alpha
    if "dispFit" in objNZ.var:
        dispFit = objNZ.var["dispFit"]
    else:
        # betaPrior routine could have been called w/o the dispersion fitted trend
        dispFit = np.mean(objNZ.var["dispersion"])

    varlogk = 1 / objNZ.var["baseMean"] + dispFit
    weights = 1 / varlogk

    if betaMatrix.shape[0] > 1:

        def _anon(x):
            # this test removes genes which have betas tending to +/- infinity
            useFinite = np.abs(x) < 10
            # if no more betas pass test, return wide prior
            if np.sum(useFinite) == 0:
                return 1e6
            else:
                if betaPriorMethod == "quantile":
                    return matchUpperQuantileForVariance(x[useFinite], upperQuantile)
                else:
                    return matchWeightedUpperQuantileForVariance(
                        x[useFinite], weights[useFinite], upperQuantile
                    )

        betaPriorVar = betaMatrix.apply(_anon, 0)

    else:
        betaPriorVar = betaMatrix**2

    # TODO superfluous, no?
    betaPriorVar.columns = betaMatrix.columns

    # intercept set to wide prior
    if "Intercept" in betaPriorVar.columns:
        betaPriorVar["Intercept"] = 1e6

    if modelMatrixType == "expanded":
        # bring over beta priors from the GLM fit without prior.
        # for factors: prior variance of each level are the average of the
        # prior variances for the levels present in the previous GLM fit
        betaPriorVar = objNZ.averagePriorsOverLevels(betaPriorVar)

    return betaPriorVar


def matchUpperQuantileForVariance(x, upperQuantile=0.05):
    sdEst = np.quantile(np.abs(x), 1 - upperQuantile) / scipy.stats.norm.ppf(
        1 - upperQuantile / 2
    )
    return sdEst**2


def matchWeightedUpperQuantileForVariance(x, weights, upperQuantile=0.05):
    sdEst = wtd_quantile(
        np.abs(x), weights=weights, probs=(1 - upperQuantile), normwt=True
    ) / scipy.stats.norm.ppf(1 - upperQuantile / 2)
    return sdEst**2
