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
import patsy

from .DESeqDataSet import DESeqDataSet, checkFullRank
from .lrt import checkLRT, nbinomLRT
from .misc import nOrMoreInCell
from .outliers import refitWithoutOutliers
from .wald import nbinomWaldTest


def DESeq(
    obj,
    test="Wald",
    fitType="parametric",
    sfType="ratio",
    betaPrior=False,
    full=None,
    reduced=None,
    quiet=False,
    minReplicatesForReplace=7,
    modelMatrixType=None,
    useT=False,
    minmu=None,
    parallel=False,
):
    """
    Differential expression analysis based on the Negative Binomial distribution.

    This function performs a default analysis through the steps:
    - estimation of size factors
    - estimation of dispersion
    - negative binomial GLM fitting and Wald statistics

    For complete details on each step, see the documentation of the respective functions.

    Arguments
    ---------
    obj : DESeqDataSet
        DESeqDataSet object
    test : str
        either "Wald" or "LRT", to choose between Wald significance tests or the likelihood ratio test on the difference in deviance between a full and a reduced model formula
        optional, defaults to "Wald"
    fitType : str
        either "parametric", "local", "mean" or "glmGamPoi" for the type of fitting of dispersions to the mean intensity
        optional, defaults to "parametric"
    sfType : str
        either "ratio", "poscounts" or "iterate" for the type of size factor estimation
        optional, defaults to "ratio"
    betaPrior : bool
        whether or not to put a zero-mean normal prior on the non-intercept coefficients
    full
        for test="LRT", the full model formula, which is restricted to the formula in obj.design. Alternatively, it can be a model matrix constructed by the user.
        Advanced use: specifying a model matric for full and test="Wald" is possible if betaPrior=False
        optional, defaults to False
    reduced
        for test="LRT", a reduced formula to compare agains, i.e. the full formula with the termes of interest removed.
        Alternatively, it can be a model matrix constructed by the user.
    quiet : bool
        whether to print messages at each step
    minReplicatesForReplace : int
        the minimum number of replicates required in order to use replaceOutliers on a sample.
        If there are samples with that many replicates, the model will be refit after these replacing outliers, flagged by Cook's distance.
        Set to inf in order to never replace outliers.
        It is set to inf if fitType="glmGamPoi"
    modelMatrixType : str
        either "standard" or "expanded". Describes how the model matrix X of the GLM formula is formed.
        If "standard", model matrix is built directly from the design formula.
        If "expanded", model matrix includes an indicator variable for each level of factors in addition to an intercept.
        betaPrior must be True in order for expanded model matrices to be fitted.
    useT : bool
        optional, defaults to False
        passed to nbinomWaldTest, where Wald statistics are assumed to follow a standard Normal
    minmu
        lower bound on the estimated count for fitting gene-wise dispersion and for use with nbinomWaldTest and nbinomLRT.
        If fitType == "glmGamPoi" then defaults to 1e-6 (as this fitType is optimized for single cell data, for which a lower minmu is recommended), otherwise defaults to 0.5
    parallel : bool
    """

    # Default values
    if minmu is None:
        if fitType == "glmGamPoi":
            minmu = 1e-6
        else:
            minmu = 0.5

    # Check arguments
    if not isinstance(obj, DESeqDataSet):
        raise ValueError("obj is not of type DESeqDataSet")

    if test not in ["Wald", "LRT"]:
        raise ValueError("invalid value for parameter test. Must be either Wald or LRT")

    if fitType not in ["parametric", "local", "mean", "glmGamPoi"]:
        raise ValueError(f"invalid value for parameter fitType: {fitType}")

    dispersionEstimator = "glmGamPoi" if fitType == "glmGamPoi" else "DESeq2"

    if not isinstance(quiet, bool):
        raise ValueError(f"invalid value for parameter quiet: {quiet}")
    if not quiet:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    if not isinstance(parallel, bool):
        raise ValueError(f"invalid value for parameter parallel: {parallel}")

    # turn off outlier replacement for glmGamPoi
    if fitType == "glmGamPoi":
        minReplicatesForReplace = np.inf
        if parallel:
            logging.warnings.warn(
                "parallelization of DESeq() is not implemented for fitType='glmGamPoi'"
            )

    if sfType not in ["ratio", "poscounts", "iterate"]:
        raise ValueError(f"invalid value for parameter 'sfType': {sfType}")

    # more argument checking
    # TODO check that minReplicatesForReplace is numeric

    modelAsFormula = not isinstance(full, (np.ndarray, pd.DataFrame))

    if not isinstance(betaPrior, bool):
        raise ValueError(f"invalid value for parameter betaPrior: {betaPrior}")

    if full is None:
        full = obj.design
    else:
        full = patsy.dmatrix(full, data=obj.obs, NA_action="raise")

    if test == "LRT":
        if reduced is None:
            raise ValueError("likelihood ratio test requires a 'reduced' design")
        if betaPrior:
            raise ValueError(
                "test='LRT' does not support use of LFC shrinkage, use betaPrior=False"
            )
        if modelMatrixType is not None and modelMatrixType == "expanded":
            raise ValueError("test='LRT' does not support use of expanded model matrix")

        if modelAsFormula == isinstance(reduced, (np.ndarray, pd.DataFrame)):
            raise ValueError(
                "if one of 'full' or 'reduced' is a matrix, the other must also be a matrix"
            )

        if modelAsFormula:
            reduced = patsy.dmatrix(reduced, data=obj.obs, NA_action="raise")
            checkLRT(full, reduced)
        else:
            checkFullRank(full)
            checkFullRank(reduced)
            if full.shape[1] <= reduced.shape[1]:
                raise ValueError(
                    "the number of columns of 'full' should be larger than the number of columns of 'reduced'"
                )

    if test == "Wald" and reduced is not None:
        raise ValueError("'reduced' ignored when test='Wald'")
    if dispersionEstimator == "glmGamPoi" and test == "Wald":
        logging.warnings.warn(
            "glmGamPoi dispersion estimator should be used in combination with a LRT and not a Wald test"
        )

    if modelAsFormula:
        # run some tests common to DESeq, nbinomWaldTest, nbinomLRT
        obj.designAndArgChecker(betaPrior)

        # warn if the design is just an intercept
        if [len(t.factors) for t in obj.design.design_info.terms] == [0]:
            logging.warnings.warn(
                "the design is ~1 (just an intercept). Is this intended?"
            )

        if full.design_info.describe() != obj.design.design_info.describe():
            raise ValueError("'full' specified as formula should match obj.design")

        modelMatrix = None

    else:
        # model not as formula, so DESeq() is using supplied model matrix
        logging.info("using supplied model matrix")
        if betaPrior:
            raise ValueError(
                "'betaPrior'=True is not supported for user-provided model matrices"
            )
        checkFullRank(full)
        # this will be used for dispersion estimation and testing
        modelMatrix = full

    obj.betaPrior = betaPrior

    if obj.normalizationFactors is not None:
        logging.info("using pre-existing normalization factors")
    elif obj.sizeFactors is not None:
        logging.info("using pre-existing size factors")
    else:
        logging.info("estimating size factors")
        obj = obj.estimateSizeFactors(type_=sfType, quiet=quiet)

    if not parallel:
        logging.info("estimating dispersions")

        obj = obj.estimateDispersions(
            fitType=fitType, quiet=quiet, modelMatrix=modelMatrix, minmu=minmu
        )

        logging.info("fitting model and testing")

        if test == "Wald":
            obj = nbinomWaldTest(
                obj,
                betaPrior=betaPrior,
                quiet=quiet,
                modelMatrix=modelMatrix,
                modelMatrixType=modelMatrixType,
                useT=useT,
                minmu=minmu,
            )
        elif test == "LRT":
            obj = nbinomLRT(
                obj,
                full=full,
                reduced=reduced,
                quiet=quiet,
                minmu=minmu,
                type_=dispersionEstimator,
            )

    else:  # if parallel
        if modelMatrixType is not None:
            if betaPrior and modelMatrixType != "expanded":
                raise ValueError(
                    "parallelization not implemented for non-expanded matrix with beta priors"
                )

        obj = DESeqParallel(
            obj,
            test=test,
            fitType=fitType,
            betaPrior=betaPrior,
            full=full,
            reduced=reduced,
            quiet=quiet,
            modelMatrix=modelMatrix,
            useT=useT,
            minmu=minmu,
            BPPARAM=BPPARAM,
        )

    # if there are sufficient replicates, then pass through to refitting function
    sufficientReps = nOrMoreInCell(obj.modelMatrix, minReplicatesForReplace).any()
    if sufficientReps:
        obj = refitWithoutOutliers(
            obj,
            test=test,
            betaPrior=betaPrior,
            full=full,
            reduced=reduced,
            quiet=quiet,
            minReplicatesForReplace=minReplicatesForReplace,
            modelMatrix=modelMatrix,
            modelMatrixType=modelMatrixType,
        )

    # TODO R DESeq2 stores the package version in obj

    return obj
