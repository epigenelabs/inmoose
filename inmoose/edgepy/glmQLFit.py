# -----------------------------------------------------------------------------
# Copyright (C) 2008-2022 Yunshun Chen, Aaron TL Lun, Davis J McCarthy, Matthew E Ritchie, Belinda Phipson, Yifang Hu, Xiaobei Zhou, Mark D Robinson, Gordon K Smyth
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

# This file is based on the files 'R/glmQLFTest.R' and 'src/R_check_poisson_bound.cpp' of the Bioconductor edgeR package (version 3.38.4).

import numpy as np
import patsy
import scipy

from ..limma import squeezeVar
from .aveLogCPM import aveLogCPM
from .DGEGLM import DGEGLM
from .glmFit import glmFit, glmLRT
from .residDF import _residDF


def glmQLFit_DGEList(
    self,
    design=None,
    dispersion=None,
    abundance_trend=True,
    robust=False,
    winsor_tail_p=(0.05, 0.1),
):
    """
    Fit a quasi-likelihood negative binomial generalized log-linear model to count data.

    See also
    --------
    glmQLFit

    Arguments
    ---------
    design : matrix, optional
        design matrix for the genewise linear models
    dispersion : float or array_like
        scalar, vector or matrix of negative binomial dispersions. If
        :code:`None`, it will be extracted from the :class:`~DGEList` object,
        with order of precedence: trended dispersions, common dispersion, a
        constant value of 0.05.
    abundance_trend : bool
        whether to allow an abundance-dependent trend when estimating the prior
        values for the quasi-likelihood multiplicative dispersion parameter.
    robust : bool
        whether to estimate the prior QL dispersion distribution robustly
    winsor_tail_p : pair of floats
        pair of floats giving the proportion to trim (Winsorize) from lower and
        upper tail of the distribution of genewise deviances when estimating
        the hyperparameters. Positive values produce robust empirical Bayes
        ignoring outlier small or large deviances. Only used when
        :code:`robust=True`.

    Returns
    -------
    DGEGLM
        object with the same components as produced by :func:`glmFit`, plus:

        - :code:`df_residual_zeros`, an array containing the number of
          effective residual degrees of freedom for each gene, taking into
          account any treatment groups with all zero counts.
        - :code:`df_prior`, a float (if :code:`robust=False`) or array (if
          :code:`robust=True`), giving the prior degrees of freedom for the QL
          dispersions.
        - :code:`var_prior`, a float (if :code:`robust=False`) or array (if
          :code:`robust=True`), giving the location of the prior distrbution
          for the QL dispersions.
        - :code:`var_post`, an array containing the posterior empirical Bayes
          QL dispersions.
    """

    if design is None:
        design = self.design
        if design is None:
            design = patsy.dmatrix("~C(self.samples['group'])")

    if dispersion is None:
        dispersion = self.trended_dispersion
        if dispersion is None:
            dispersion = self.common_dispersion
        if dispersion is None:
            raise ValueError("No dispersion values found in DGEList object")

    offset = self.getOffset()
    if self.AveLogCPM is None:
        self.AveLogCPM = self.aveLogCPM()

    fit = glmQLFit(
        y=self.counts,
        design=design,
        dispersion=dispersion,
        offset=offset,
        lib_size=None,
        abundance_trend=abundance_trend,
        AveLogCPM=self.AveLogCPM,
        robust=robust,
        winsor_tail_p=winsor_tail_p,
        weights=self.weights,
    )
    fit.samples = self.samples
    fit.genes = self.genes
    fit.AveLogCPM = self.AveLogCPM
    return fit


def glmQLFit(
    y,
    design=None,
    dispersion=None,
    offset=None,
    lib_size=None,
    weights=None,
    abundance_trend=True,
    AveLogCPM=None,
    robust=False,
    winsor_tail_p=(0.05, 0.1),
):
    """
    Fit a quasi-likelihood negative binomial generalized log-linear model to count data.

    Implement one of the quasi-likelihood (QL) methods of [Lund2012]_, with some
    enhancements and with slightly different glm, trend and FDR methods. See
    [Lun2016]_ or [Chen2016]_ for tutorials describing the use of
    :func:`glmQLFit` and :func:`glmQLFTest` as part of a complete pipeline.
    Another case study using :func:`glmQLFit` and :func:`glmQLFTest` is given in
    Section 4.7 of the edgeR User's Guide.

    :func:`glmQLFit` is similar to :func:`glmFit` except that it also estimates
    QL dispersion values. It calls the limma function :func:`.squeezeVar` to
    conduct empirical Bayes moderation of the genewise QL dispersions. If
    :code:`robust=True`, then the robust hyperparameter estimation features of
    :func:`.squeezeVar` are used [Phipson2016]_. If
    :code:`abundance_trend=True`, then a prior trend is estimated based on the
    average logCPMs.

    :func:`glmQLFit` gives special attention to handling of zero counts, and in
    particular to situations when fitted values of zero provide no useful
    residual degrees of freedom for estimating the QL dispersion [Lun2017]_.
    The usual residual degrees of freedom are returned as :code:`df_residual`
    while the adjusted residual degrees of freedom are returned as
    :code:`df_residual_zeros`.

    Note
    ----
    The negative binomial dispersions :code:`dispersion` supplied to
    :func:`glmQLFit` and :func:`glmQLFTest` must be based on a global model, that
    is, they must be either trended or common dispersions. It is not correct to
    supply genewise dispersions because :func:`glmQLFTest` estimates genewise
    variability using the QL dispersion.

    Arguments
    ---------
    y : matrix
        a matrix of counts
    design : matrix, optional
        design matrix for the genewise linear models
    dispersion : float or array_like
        scalar, vector or matrix of negative binomial dispersions. If
        :code:`None`, it will be extracted from the :class:`~DGEList` object,
        with order of precedence: trended dispersions, common dispersion, a
        constant value of 0.05.
    abundance_trend : bool
        whether to allow an abundance-dependent trend when estimating the prior
        values for the quasi-likelihood multiplicative dispersion parameter.
    robust : bool
        whether to estimate the prior QL dispersion distribution robustly
    winsor_tail_p : pair of floats
        pair of floats giving the proportion to trim (Winsorize) from lower and
        upper tail of the distribution of genewise deviances when estimating
        the hyperparameters. Positive values produce robust empirical Bayes
        ignoring outlier small or large deviances. Only used when
        :code:`robust=True`.

    Returns
    -------
    DGEGLM
        object with the same components as produced by :func:`glmFit`, plus:

        - :code:`df_residual_zeros`, an array containing the number of
          effective residual degrees of freedom for each gene, taking into
          account any treatment groups with all zero counts.
        - :code:`df_prior`, a float (if :code:`robust=False`) or array (if
          :code:`robust=True`), giving the prior degrees of freedom for the QL
          dispersions.
        - :code:`var_prior`, a float (if :code:`robust=False`) or array (if
          :code:`robust=True`), giving the location of the prior distrbution
          for the QL dispersions.
        - :code:`var_post`, an array containing the posterior empirical Bayes
          QL dispersions.
    """
    glmfit = glmFit(
        y,
        design=design,
        dispersion=dispersion,
        offset=offset,
        lib_size=lib_size,
        weights=weights,
    )

    # Setting up the abundances.
    if abundance_trend:
        if AveLogCPM is None:
            AveLogCPM = aveLogCPM(
                y, lib_size=lib_size, weights=weights, dispersion=dispersion
            )
        glmfit.AveLogCPM = AveLogCPM
    else:
        AveLogCPM = None

    # Adjust df_residual for fitted values at zero
    zerofit = (glmfit.fitted_values < 1e-4) & (glmfit.counts < 1e-4)
    df_residual = _residDF(zerofit, glmfit.design)

    # Empirical Bayes squeezing of the quasi-likelihood variance factors
    with np.errstate(invalid="ignore"):
        s2 = glmfit.deviance / df_residual
    s2[df_residual == 0] = 0
    s2 = np.maximum(s2, 0)
    s2_fit = squeezeVar(
        s2,
        df=df_residual,
        covariate=AveLogCPM,
        robust=robust,
        winsor_tail_p=winsor_tail_p,
    )

    # Storing results
    glmfit.df_residual_zeros = df_residual
    glmfit.df_prior = s2_fit["df_prior"]
    glmfit.var_post = s2_fit["var_post"]
    glmfit.var_prior = s2_fit["var_prior"]
    return glmfit


def glmQLFTest(glmfit, coef=None, contrast=None, poisson_bound=True):
    """
    Conduct genewise statistical tests for a given coefficient or contrast

    Implement one of the quasi-likelihood (QL) methods of [Lund2012]_, with some
    enhancements and with slightly different glm, trend and FDR methods. See
    [Lun2016]_ or [Chen2016]_ for tutorials describing the use of
    :func:`glmQLFit` and :func:`glmQLFTest` as part of a complete pipeline.
    Another case study using :func:`glmQLFit` and :func:`glmQLFTest` is given in
    Section 4.7 of the edgeR User's Guide.

    :func:`glmQLFTest` is similar to :func:`glmLRT` except that it replaces
    likelihood ratio tests with empirical Bayes quasi-likelihood F-tests. The
    *p*-values from :func:`glmQLFTest` are always greater than or equal to those
    that would be obtained from :func:`glmLRT` using the same negative binomial
    dispersions.

    Note
    ----
    The negative binomial dispersions :code:`dispersion` supplied to
    :func:`glmQLFit` and :func:`glmQLFTest` must be based on a global model, that
    is, they must be either trended or common dispersions. It is not correct to
    supply genewise dispersions because :func:`glmQLFTest` estimates genewise
    variability using the QL dispersion.

    Arguments
    ---------
    glmfit : DGEGLM
        a :class:`DGEGLM` object, usually output from :func:`glmQLFit`
    coeff : int or string array
        indicated which coefficients of the linear model are to be tested equal to zero.
        Ignored if :code:`contrast` is not :code:`None`.
    contrast : array_like
        vector or matrix specifying one or more contrasts of the linear model
        coefficients to be tested equal to zero.
    poisson_bound : bool
        if :code:`True` then the *p*-value returned will never be less than
        would be obtained for a likelihood ratio test with NB dispersion equal
        to zero.

    Returns
    -------
    DGELRT
        an object of class :class:`DGELRT` with the same components as produced
        by :func:`glmLRT`, except that the :code:`"stat"` column of the
        :code:`table` contains quasi-likelihood F-statistics. It also stored
        :code:`df_total`, an array containing the denominator degrees of
        freedom for the F-test, equal to :code:`df_prior + df_residual_zeros`.
    """

    if coef is None:
        coef = glmfit.design.shape[1] - 1

    if not isinstance(glmfit, DGEGLM):
        raise ValueError("glmfit must be a DGEGLM object produced by glmQLFit")
    if glmfit.var_post is None:
        raise ValueError("need to run glmQLFit before glmQLFTest")
    out = glmLRT(glmfit, coef=coef, contrast=contrast)

    # compute the QL F-statistic
    F_stat = out["stat"] / out.df_test / glmfit.var_post
    df_total = glmfit.df_prior + glmfit.df_residual_zeros
    max_df_residual = glmfit.counts.shape[1] - glmfit.design.shape[1]
    df_total = np.minimum(df_total, glmfit.counts.shape[0] * max_df_residual)

    # compute p-values from the QL F-statistic
    F_pvalue = scipy.stats.f.sf(F_stat, dfn=out.df_test, dfd=df_total)

    # Ensure is not more significant than chisquare test with Poisson variance
    if poisson_bound:
        i = _isBelowPoissonBound(glmfit)
        if i.any():
            pois_fit = glmFit(
                glmfit.counts.loc[i, :],
                design=glmfit.design,
                offset=np.broadcast_to(glmfit.offset, glmfit.counts.shape)[i, :],
                weights=glmfit.weights[i, :] if glmfit.weights is not None else None,
                start=glmfit.unshrunk_coefficients[i],
                dispersion=0,
            )
            pois_res = glmLRT(pois_fit, coef=coef, contrast=contrast)
            F_pvalue[i] = np.maximum(F_pvalue[i], pois_res["pvalue"])

    out["stat"] = F_stat
    out["pvalue"] = F_pvalue
    out.df_total = df_total

    return out


def _isBelowPoissonBound(glmfit):
    """a convenience function"""
    disp = glmfit.dispersion
    s2 = glmfit.var_post[:, None]
    fitted = glmfit.fitted_values

    return (((fitted * disp + 1) * s2) < 1).any(axis=1)


def plotQLDisp(
    glmfit,
    xlab="Average Log2 CPM",
    ylab="Quarter-Root Mean Deviance",
    col_shrunk="red",
    col_trend="blue",
    col_raw="black",
):
    """
    Plot the genewise quasi-likelihood dispersion against the gene abundance (in log2 counts per million)

    This function displays the quarter-root of the quasi-likelihood dispersions
    for all genes, before and after shrinkage towards a trend. If
    :code:`glmfit` was constructed without an abundance trend, the function
    instead plots a horizontal line (of color :code:`col_trend`) at the common
    value towards which dispersions are shrunk. The quarter-root transformation
    is applied to improve visibility for dispersions around unity.

    Arguments
    ---------
    glmfit : DGEGLM
        a DGEGLM object produced by :func:`glmQLFit`
    xlab : str
        label for the x-axis
    ylab : str
        label for the y-axis
    col_shrunk : str
        color of the points representing the squeezed quasi-likelihood dispersions
    col_trend : str
        color of the line showing dispersion trend
    col_raw : str
        color of the line showing the unshrunk dispersions
    """
    import matplotlib.pyplot as plt

    A = glmfit.AveLogCPM
    if A is None:
        A = glmfit.aveLogCPM()
    A = np.asarray(A)
    s2 = glmfit.deviance / glmfit.df_residual_zeros
    if glmfit.var_post is None:
        raise ValueError("need to run glmQLFit before plotQLDisp")

    plt.scatter(A, np.sqrt(np.sqrt(s2)), c=col_raw, label="Raw")
    plt.scatter(A, np.sqrt(np.sqrt(glmfit.var_post)), c=col_shrunk, label="Shrunk")
    if isinstance(glmfit.var_prior, float):
        plt.axhline(y=np.sqrt(np.sqrt(glmfit.var_prior)), c=col_trend, label="Trend")
    else:
        o = np.argsort(A)
        plt.plot(
            A[o], np.sqrt(np.sqrt(glmfit.var_prior[o])), c=col_trend, label="Trend"
        )

    plt.legend(loc="upper right")
    plt.show()
