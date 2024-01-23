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

# This file is based on the file 'R/squeezeVar.R' of the Bioconductor limma package (version 3.55.1).


import numpy as np

from .fitFDist import fitFDist


def squeezeVar(var, df, covariate=None, robust=False, winsor_tail_p=(0.05, 0.1)):
    """
    Squeeze a set of sample variances together by computing empirical Bayes posterior means
    This function implements empirical Bayes algorithms proposed by
    [Smyth2004]_ and [Phipson2016]_.

    A conjugate Bayesian hierarchical model is assumed for a set of sample
    variances. The hyperparameters are estimated by fitting a scaled
    F-distribution to the sample variances. The function returns the posterior
    variances and the estimated hyperparameters.

    Specifically, the sample variances :code:`var` are assumed to follow scaled
    chi-squared distributions, conditional on the true variances, and a scaled
    inverse chi-squared prior is assumed for the true variances.  The scale and
    degrees of freedom of this prior distribution are estimated from the values
    of :code:`var`.

    The effect of this function is to squeeze the variances towards a common
    value, or to a global trend if a :code:`covariate` is provided. The
    squeezed variances have a smaller expected mean square error to the true
    variances than do the sample variances themselves.

    If :code:`covariate` is not :code:`None`, then the scale parameter of the
    prior distribution is assumed to depend on the covariate. If the covariate
    is average log-expression, then the effect is an intensity-dependent trend
    similar to that in [Sartor2006]_.

    :code:`robust=True` implements the robust empirical Bayes procedure of
    [Phipson2016]_ which allows some of the :code:`var` values to be outliers.

    Arguments
    ---------
    var : array_like
        1-D array of independent sample variances
    df : array_like
        1-D array of degrees of freedom for the sample variances
    covariate : ??
        if not :code:`None`, :code:`var_prior` will depend on this numeric
        covariate. Otherwise, :code:`var_prior` is constant.
    robust : bool
        whether the estimation of :code:`df_prior` and :code:`var_prior` be
        robustified against outlier sample variances
    winsor_tail_p : float or pair of floats
        left and right tail proportions of :code:`x` to Winsorize. Only used
        when :code:`robust=True`

    Returns
    -------
    dict
        a dictionnary with keys:

        - :code:`"var_post"`, 1-D array of posterior variances. Same length as
          :code:`var`.
        - :code:`"var_prior"`, location or scale of prior distribution. 1-D
          array of same length as :code:`var` if :code:`covariate` is not
          :code:`None`, otherwise a single value.
        - :code:`"df_prior"`, degrees of freedom of prior distribution. 1-D
          array of same length as :code:`var` if :code:`robust=True`, otherwise
          a single value.
    """
    n = len(var)

    # Degenerate special cases
    if n == 0:
        raise ValueError("var is empty")
    if n == 1:
        return {"var_post": var, "var_prior": var, "df_prior": 0}

    # When df==0, guard against missing or infinite values in var
    df = np.broadcast_to(df, var.shape)
    var[df == 0] = 0

    # Estimate hyperparameters
    if robust:
        raise NotImplementedError("Robust estimation in squeezeVar is not implemented")
    else:
        fit = fitFDist(var, df1=df, covariate=covariate)
        df_prior = fit["df2"]
    if np.isnan(df_prior).any():
        raise RuntimeError("Could not estimate prior df")

    # Posterior variances
    var_post = _squeezeVar(var=var, df=df, var_prior=fit["scale"], df_prior=df_prior)

    return {"var_post": var_post, "var_prior": fit["scale"], "df_prior": df_prior}


def _squeezeVar(var, df, var_prior, df_prior):
    """
    Squeeze posterior variances given hyperparameters

    NAs not allowed in :code:`df_prior`

    Arguments
    ---------
    var : array_like
        1-D array of independent sample variances
    df : array_like
        1-D array of degrees of freedom for the sample variances
    var_prior : array_like
        array of prior variances
    df_prior :
        array of degrees of freedom for the prior variances

    Returns
    -------
    ndarray
        array of posterior variances
    """
    df = np.broadcast_to(df, var.shape)
    var_prior = np.broadcast_to(var_prior, var.shape)
    df_prior = np.broadcast_to(df_prior, var.shape)

    isfin = np.isfinite(df_prior)
    var_post = np.zeros(var.shape)

    # For infinite df_prior, return var_prior
    var_post[~isfin] = var_prior[~isfin]

    var_post[isfin] = (df[isfin] * var[isfin] + df_prior[isfin] * var_prior[isfin]) / (
        df[isfin] + df_prior[isfin]
    )

    return var_post
