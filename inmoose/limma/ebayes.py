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


import numpy as np
import scipy


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
