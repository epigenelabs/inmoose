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


def nbinomLRT(
    obj,
    full=None,
    reduced=None,
    betaTol=1e-8,
    maxit=100,
    useOptim=True,
    quiet=False,
    useQR=True,
    minmu=None,
    type_="DESeq2",
):
    """
    Likelihood ratio test (chi-squared test) for GLMs

    This function tests for significance of change in deviance between a full
    and reduced model which are provided as :code:`formula`. Fitting uses
    previously calculated :attr:`.DESeqDataSet.sizeFactors` (or
    :attr:`.DESeqDataSet.normalizationFactors`) and dispersion estimates.

    The difference in deviance is compared to chi-squared distribution with df
    = (reduced residual degrees of freedom - full residual degrees of freedom).
    This function is an alternative to :func:`nbinomWaldTest`.

    See Also
    --------
    DESeq
    nbinomWaldTest

    Arguments
    ---------
    obj : DESeqDataSet
        a DESeqDataSet
    full
        the full model formula, this should be the formula in
        :code:`obj.design`.  Alternatively can be a matrix.
    reduced
        a reduced formula to compare against, e.g. the full model with a term
        or terms of interest removed.
    betaTol : float
        control parameter defining convergence
    maxit : int
        the maximum number of iterations to allow for convergence of the
        coefficient vector
    useOptim : bool
        whether to use the native optim function on columns which do not
        converge within :code:`maxit` iterations
    quiet : bool
        whether to print messages at each step
    useQR : bool
        whether to use the QR decomposition of the design matrix while fitting
        the GLM
    minmu : float
        lower bound on the estimated count while fitting the GLM
    type_ : "DESeq2" or "glmGamPoi"
        If :code:`"DESeq2"`, a classical Likelihood ratio test based on the Chi-squared distribution is conducted.

        If :code:`"glmGamPoi"` and previously the dispersion has been estimated
        with :code:`"glmGamPoi"` as well, a quasi-likelihood ratio test based
        on the F-distribution is conducted. It is supposed to be more accurate,
        because it takes the uncertainty of dispersion estimate into account
        in the same way that a t-test improves upon a Z-test.

    Returns
    -------
    DESeqDataSet
        the input :code:`obj` with new results columns accessible through
        :meth:`.DESeqDataSet.results`. The coefficients and standard errors are
        reported on a log2 scale.
    """

    if type_ not in ["DESeq2", "glmGamPoi"]:
        raise ValueError(f"invalid value for type_: {type_}")
    if "dispersion" not in obj.var:
        raise ValueError(
            "testing requires dispersion estimates, first call estimateDispersions()"
        )

    if reduced is None:
        raise ValueError(
            'provide a reduced formula for the LRT, e.g. nbinomLRT(obj, reduced="~1")'
        )

    # run check on the formula
    raise NotImplementedError()


def checkLRT(full, reduced):
    """check for LRT formulas, written as function to share code between DESeq and nbinomLRT"""
    reducedNotInFull = ~np.isin(reduced.design_info.terms, full.design_info.terms)
    if np.any(reducedNotInFull):
        reducedVars = [
            v for (v, x) in zip(reduced.design_info.term_names, reducedNotInFull) if x
        ]
        raise ValueError(
            f"the following variables in the reduced formula not in the full formula: {' '.join(reducedVars)}"
        )

    full.design_info.terms
