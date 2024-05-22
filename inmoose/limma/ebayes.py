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

# This file is based on the file 'R/ebayes.R' of the Bioconductor limma package (version 3.55.1).


import logging

import numpy as np
import pandas as pd
import scipy

from .decidetests import classifyTestsF
from .marraylm import MArrayLM
from .squeezeVar import squeezeVar


def eBayes(
    fit,
    proportion=0.01,
    stdev_coef_lim=(0.1, 4),
    trend=False,
    robust=False,
    winsor_tail_p=(0.05, 0.1),
):
    """
    Empirical Bayes statistics for differential expression

    Given a linear model fit from :func:`lmFit`, compute moderated
    *t*-statistics, moderated *F*-statistics, and log-odds of differential
    expression by empirical Bayes moderation of the standard errors towards a
    global value.

    This function is used to rank genes in order of evidence for differential
    expression. They use an empirical Bayes method to squeeze the genewise
    residual variances towards a common value (or towards a global trend)
    [Smyth2004]_ [Phipson2016]_.  The degrees of freedom for the individual
    variances are increased to reflect the extra information gained from the
    empirical Bayes moderation, resulting in increased statistical power to
    detect differential expression.

    This function accepts as input a :class:`MArrayLM` fitted model :code:`fit`
    produced by :func:`lmFit`. The columns of :code:`fit` define a set of
    contrasts which are to be tested equal to zero. The fitted model object may
    have been processed by :func:`contrasts_fit` before being passed to
    :func:`eBayes` to convert the coefficients of the original design matrix
    into an arbitrary number of contrasts.

    The empirical Bayes moderated *t*-statistics test each individual contrast
    equal to zero. For each gene (row), the moderated *F*-statistic tests
    whether all the contrasts are zero. The *F*-statistic is an overall test
    computed from the set of *t*-statistics for that probe. This is exactly
    analogous to the relationship between *t*-tests and *F*-statistics in
    conventional ANOVA, except that the residual mean squares have been
    moderated between genes.

    The estimates :code:`s2_prior` and :code:`df_prior` are computed by
    :func:`fitFDist`. :code:`s2_post` is the weighted average of
    :code:`s2_prior` and :code:`sigma**2` with weights proportional to
    :code:`df_prior` and :code:`df_residual` respectively. The log-odds of
    differential expression :code:`lods` was called the *B-statistic* by
    [Loennstedt2002]_. The *F*-statistics :code:`F` are computed by
    :func:`classifyTestsF` with :code:`fstats_only=True`.

    :func:`eBayes` does not compute ordinary *t*-statistics because they always
    have worse performance than the moderated versions. The ordinary
    (unmoderated) *t*-statistics can, however, be easily extracted from the
    linear model output for comparison purposes -- see the example code below.

    The use of :func:`eBayes` with :code:`trend=True` is known as the
    *limma-trend* method [Law2014]_ [Phipson2016]_. With this option, an
    intensity-dependent trend is fitted to the prior variances
    :code:`s2_prior`. Specifically, :func:`squeezeVar` is called with the
    :code:`covariate` equal to :code:`Amean`, the average log2-intensity for
    each gene. The trend that is fitted can be examined by :func:`plotSA`.
    limma-trend is useful for processing expression values that show a
    mean-variance relationship. This is often useful for microarray data, and
    it can also be applied to RNA-Seq counts that have been converged to
    log2-counts per million (logCPM) values [Law2014]_. When applied to RNA-Seq
    logCPM values, limma-trend give similar results to the :func:`voom` method.
    The voom method incorporates the mean-variance trend into the precision
    weights, whereas limma-trend incorporates the trend into the empirical
    Bayed moderation. limma-trend is somewhat simpler than :code:`voom` because
    it assumes that the sequencing depths (library sizes) are wildly different
    between the samples and it applies the mean-variance trend on a genewise
    basis instead to individual observations. limma-trend is recommended for
    RNA-Seq analysis when the library sizes are reasonably consistent (less
    than 3-fold difference from smallest to largest) because of its simplicity
    and speed.

    If :code:`robust=True`, then the robust empirical Bayes procedure of
    [Phipson2016]_ is used. This is frequently useful to protect the empirical
    Bayes procedure agains hyper-variable or hypo-variable genes, especially
    when analysing RNA-Seq data. See :func:`squeezeVar` for more details.

    Arguments
    ---------
    fit : MArrayLM
        fitted model object produce by :func:`lmFit` or :func:`contrasts_fit`
    proportion : float
        value between 0 and 1, assumed proportion of genes which are
        differentially expressed
    stdev_coef_lim : pair of floats
        assumed lower and upper limits for the standard deviations of
        log2-fold-changes for differentially expressed genes
    trend : bool or array_like
        whether an intensity-dependent trend should be allowed for the prior
        variance.  If :code:`False` then the prior variance is constant.
        Alternatively, :code:`trend` can be a row-wise array, which will be used
        as the covariate for the prior variance.
    robust : bool
        whether the estimation of :code:`df_prior` and :code:`var_prior` should
        be robustified against outlier sample variances.

    Returns
    -------
    MArrayLM
        an object containing everything found in :code:`fit` plus the following
        added attributes:

        - :code:`t`: matrix of moderated *t*-statistics
        - :code:`p_value`: matrix of two-sided *p*-values corresponding to the
          *t*-statistics
        - :code:`lods` matrix giving the log-odds of differential expression
          (on the natural log scale)
        - :code:`s2_prior`: estimated prior value for :code:`sigma**2`. A
          row-wise array if :code:`covariate` is non-:code:`None`, otherwise a
          single value
        - :code:`df_prior`: degrees of freedom associated with
          :code:`s2_prior`. A row-wise array if :code:`robust=True`, otherwise
          a single value
        - :code:`df_total`: row-wise array giving the total degrees of freedom
          associated with the *t*-statistics for each gene. Equal to
          :code:`df_prior + df_residual` or :code:`sum(df_residual)`, whichever
          is smaller.
        - :code:`s2_post`: row-wise array giving the posterior values for
          :code:`sigma**2`
        - :code:`var_prior`: column-wise array giving estimated prior values for
          the variance of the log2-fold-changes for differentially expressed
          gene for each contrast. Used for evaluating :code:`lods`.
        - :code:`F`: row-wise array of moderated *F*-statistics for testing all
          contrasts defined by the columns of :code:`fit` simultaneously equal
          to zero.
        - :code:`F_p_value`: row-wise array giving *p*-values corresponding to
          :code:`F`

        The matrices :code:`t`, :code:`p_value`, :code:`lods` have the same
        dimensions as the input object :code:`fit`, with rows corresponding to
        genes and columns to coefficients or constrats.  The vector
        :code:`s2_prior`, :code:`df_prior`, :code:`df_total`, :code:`F` and
        :code:`F_p_value` correspond to rows, with length equal to the number
        of genes. The vector :code:`var_prior` corresponds to columns, with
        length equal to the number of contrasts. If :code:`s2_prior` or
        :code:`df_prior` have length 1, then the same value applies to all
        genes.

        :code:`s2_prior`, :code:`df_prior` and :code:`var_prior` contain
        empirical Bayes hyperparameters used to obtain :code:`df_total`,
        :code:`s2_post` and :code:`lods`.
    """
    if not isinstance(fit, MArrayLM):
        raise ValueError("fit is not a valid MArrayLM object")
    if isinstance(trend, bool) and trend and (fit.Amean is None):
        raise ValueError("Need Amean component in fit to estimate trend")
    eb = _ebayes(
        fit=fit,
        proportion=proportion,
        stdev_coef_lim=stdev_coef_lim,
        trend=trend,
        robust=robust,
        winsor_tail_p=winsor_tail_p,
    )
    fit.df_prior = eb["df_prior"]
    fit.s2_prior = eb["s2_prior"]
    fit.var_prior = eb["var_prior"]
    fit.proportion = proportion
    fit.s2_post = eb["s2_post"]
    fit.t = eb["t"]
    fit.df_total = eb["df_total"]
    fit.p_value = eb["p_value"]
    fit.lods = eb["lods"]
    if (fit.design is not None) and np.linalg.matrix_rank(
        fit.design
    ) == fit.design.shape[1]:
        F_stat = classifyTestsF(fit, fstat_only=True)
        fit.F = np.asarray(F_stat)
        df1 = F_stat.df1
        df2 = F_stat.df2
        # if Y ~ F(df1, df2), then df1 Y ~ chi2(df1) when df2 -> infinity
        if df2 > 1e10:
            fit.F_p_value = scipy.stats.chi2.sf(df1 * fit.F, df=df1)
        else:
            fit.F_p_value = scipy.stats.f.sf(fit.F, dfn=df1, dfd=df2)

    return fit


def _ebayes(
    fit,
    proportion=0.01,
    stdev_coef_lim=(0.1, 4),
    trend=False,
    robust=False,
    winsor_tail_p=(0.05, 0.1),
):
    coefficients = fit.coefficients
    stdev_unscaled = fit.stdev_unscaled
    assert np.array_equal(coefficients.index, stdev_unscaled.index)
    sigma = fit.sigma
    df_residual = fit.df_residual
    if (
        (coefficients is None)
        or (stdev_unscaled is None)
        or (sigma is None)
        or (df_residual is None)
    ):
        raise ValueError("argument is not a valid lmFit object")
    if np.max(df_residual) == 0:
        raise ValueError("No residual degrees of freedom in linear model fits")
    if not np.any(np.isfinite(sigma)):
        raise ValueError("no finite residual standard deviations")
    if isinstance(trend, bool):
        if trend:
            covariate = fit.Amean
            if covariate is None:
                raise ValueError("need Amean component in fit to estimate trend")
        else:
            covariate = None
    else:
        covariate = np.broadcast_to(trend, sigma.shape)

    # Moderated t-statistic
    out = squeezeVar(
        sigma**2,
        df_residual,
        covariate=covariate,
        robust=robust,
        winsor_tail_p=winsor_tail_p,
    )
    out["s2_prior"] = out["var_prior"]
    out["s2_post"] = pd.Series(out["var_post"], index=fit.coefficients.index)
    del out["var_prior"]
    del out["var_post"]
    out["t"] = coefficients / stdev_unscaled / np.sqrt(out["s2_post"].values[:, None])
    df_total = df_residual + out["df_prior"]
    df_pooled = np.nansum(df_residual)
    df_total = np.minimum(df_total, df_pooled)
    out["df_total"] = df_total
    out["p_value"] = out["t"].apply(
        lambda x: 2 * scipy.stats.t.cdf(-np.abs(x), df=df_total), axis=0
    )

    # B-statistic
    var_prior_lim = np.array(stdev_coef_lim) ** 2 / np.median(out["s2_prior"])
    out["var_prior"] = tmixture_matrix(
        out["t"], stdev_unscaled, df_total, proportion, var_prior_lim
    )
    if np.any(np.isnan(out["var_prior"])):
        out["var_prior"][np.isnan(out["var_prior"])] = 1 / out["s2_prior"]
        logging.warnings.warn("Estimation of var_prior failed -- set to default value")

    r = np.outer(np.ones(out["t"].shape[0]), out["var_prior"])
    r = (stdev_unscaled**2 + r) / stdev_unscaled**2
    t2 = out["t"] ** 2
    Infdf = out["df_prior"] > 1e6
    if np.any(Infdf):
        kernel = t2 * (1 - 1 / r) / 2
        if np.any(~Infdf):
            t2_f = t2[~Infdf]
            r_f = r[~Infdf]
            df_total_f = df_total[~Infdf]
            kernel[~Infdf] = (
                (1 + df_total_f)
                / 2
                * np.log((t2_f + df_total_f) / (t2_f / r_f + df_total_f))
            )
    else:
        kernel = (
            (1 + df_total[:, None])
            / 2
            * np.log((t2 + df_total[:, None]) / (t2 / r + df_total[:, None]))
        )
    out["lods"] = np.log(proportion / (1 - proportion)) - np.log(r) / 2 + kernel
    return out


def tmixture_matrix(tstat, stdev_unscaled, df, proportion, v0_lim=None):
    """
    Estimate scale factor in mixture of *t*-distributions

    This function estimates the unscaled standard deviation of the true
    (unobserved) log fold changes for differentially expressed genes. It is
    used internally by :func:`eBayes` and is not intended to be called directly
    by users.

    The values in each column of :code:`tstat` are assumed to follow a mixture
    of an ordinary *t*-distribution, with mixing proportion
    :code:`1-proportion`, and :code:`(v0+v1)/v1` times a *t*-distribution, with
    mixing proportion :code:`proportion`. Here :code:`v1` is
    :code:`stdev_unscaled**2` and :code:`v0` is the value to be estimated.

    Arguments
    ---------
    tstat : ndarray
        matrix of *t*-statistics
    stdev_unscaled : ndarray
        matrix, conformal with :code:`tstat`, containing the unscaled standard
        deviations of the coefficients used to compute the *t*-statistics
    df : array_like
        array giving the degrees of freedom associated with :code:`tstat`
    proportion : float
        assumed proportion of genes that are differentially expressed
    v0_lim : pair of floats, optional
        lower and upper limits for the estimated unscaled standard deviations

    Returns
    -------
    ndarray
        estimated :code:`v0` values, one for each column of :code:`tstat`
    """
    if tstat.shape != stdev_unscaled.shape:
        raise ValueError("dimensions of tstat and stdev_unscaled do not match")
    if (tstat.shape[0],) != df.shape:
        raise ValueError("dimensions of tstat and df do not match")
    if v0_lim is not None:
        if len(v0_lim) != 2:
            raise ValueError("v0_lim must be a pair")
    ncoef = tstat.shape[1]
    if not isinstance(tstat, np.ndarray):
        tstat = np.array(tstat)
    if not isinstance(stdev_unscaled, np.ndarray):
        stdev_unscaled = np.array(stdev_unscaled)
    v0 = np.array(
        [
            tmixture_vector(tstat[:, j], stdev_unscaled[:, j], df, proportion, v0_lim)
            for j in range(ncoef)
        ]
    )
    return v0


def tmixture_vector(tstat, stdev_unscaled, df, proportion, v0_lim=None):
    """
    Estimate scale factor in mixture of *t*-distributions

    This function estimates the unscaled standard deviation of the true
    (unobserved) log fold changes for differentially expressed genes. It is
    used internally by :func:`eBayes` and is not intended to be called directly
    by users.

    The values in :code:`tstat` are assumed to follow a mixture of an ordinary
    *t*-distribution, with mixing proportion :code:`1-proportion`, and
    :code:`(v0+v1)/v1` times a *t*-distribution, with mixing proportion
    :code:`proportion`. Here :code:`v1` is :code:`stdev_unscaled**2` and
    :code:`v0` is the value to be estimated.

    Arguments
    ---------
    tstat : ndarray
        vector of *t*-statistics
    stdev_unscaled : ndarray
        vector, conformal with :code:`tstat`, containing the unscaled standard
        deviations of the coefficients used to compute the *t*-statistics
    df : array_like
        array giving the degrees of freedom associated with :code:`tstat`
    proportion : float
        assumed proportion of genes that are differentially expressed
    v0_lim : pair of floats, optional
        lower and upper limits for the estimated unscaled standard deviations

    Returns
    -------
    float
        estimated :code:`v0` value
    """
    if tstat.shape != stdev_unscaled.shape:
        raise ValueError("dimensions of tstat and stdev_unscaled do not match")
    if tstat.shape != df.shape:
        raise ValueError("dimensions of tstat and df do not match")
    tstat = np.asarray(tstat, dtype=float)
    stdev_unscaled = np.asarray(stdev_unscaled, dtype=float)
    # remove missing values
    if np.isnan(tstat).any():
        o = ~np.isnan(tstat)
        tstat = tstat[o]
        stdev_unscaled = stdev_unscaled[o]
        df = df[o]

    # ntarget t-statistics will be used for estimation
    ngenes = len(tstat)
    ntarget = int(np.ceil(proportion / 2 * ngenes))
    if ntarget < 1:
        return np.nan

    # if ntarget is very small, ensure p at least matches selected proportion
    # this ensures ptarget < 1
    p = np.max([ntarget / ngenes, proportion])

    # method requires that df be equal
    tstat = np.abs(tstat)
    MaxDF = np.max(df)
    i = df < MaxDF
    if np.any(i):
        # NB: original R code computes the proba on the log scale, but scipy
        # misses a "ilogsf" function
        TailP = scipy.stats.t.sf(tstat[i], df=df[i])
        tstat[i] = scipy.stats.t.isf(TailP, df=MaxDF)
        df[i] = MaxDF

    # Select top statistics
    o = np.flip(np.argsort(tstat))[:ntarget]
    tstat = tstat[o]
    v1 = stdev_unscaled[o] ** 2

    # compare to order statistics
    r = np.arange(ntarget) + 1.0
    p0 = 2 * scipy.stats.t.sf(tstat, df=MaxDF)
    ptarget = ((r - 0.5) / ngenes - (1 - p) * p0) / p
    v0 = np.zeros(ntarget)
    pos = ptarget > p0
    if pos.any():
        qtarget = scipy.stats.t.isf(ptarget[pos] / 2, df=MaxDF)
        v0[pos] = v1[pos] * ((tstat[pos] / qtarget) ** 2 - 1)
    if v0_lim is not None:
        v0 = np.clip(v0, v0_lim[0], v0_lim[1])
    return np.mean(v0)
