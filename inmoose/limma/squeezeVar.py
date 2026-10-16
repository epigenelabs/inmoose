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

# Based on limma's R/squeezeVar.R, including the 12 July 2025 estimator selection.


import numpy as np

from .fitFDist import fitFDist
from .fitFDistUnequalDF1 import fitFDistUnequalDF1


def squeezeVar(
    var,
    df,
    covariate=None,
    robust=False,
    winsor_tail_p=(0.05, 0.1),
    legacy=None,
    span=None,
):
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

    With unequal positive residual degrees of freedom, the default estimator
    is :func:`fitFDistUnequalDF1`, as in limma since version 3.61.8. It uses
    weighted log-scale moments and profile likelihood. Equal positive df use
    the legacy moment estimator :func:`fitFDist`. Fewer than three variances
    are returned unchanged, with zero prior df.

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
        Robust estimation is not implemented; True raises NotImplementedError
        when hyperparameter estimation is needed.
    winsor_tail_p : float or pair of floats
        left and right tail proportions of :code:`x` to Winsorize. Only used
        when :code:`robust=True`
    legacy : bool or None
        True selects :func:`fitFDist`; False selects :func:`fitFDistUnequalDF1`.
        None selects the legacy estimator if all positive df are equal.
    span : float or None
        LOWESS span for the new estimator's covariate trend. Supplying a span
        selects the new estimator even if :code:`legacy=True`, as in R limma.

    Returns
    -------
    dict
        a dictionary with keys:

        - :code:`"var_post"`, 1-D array of posterior variances. Same length as
          :code:`var`.
        - :code:`"var_prior"`, location or scale of prior distribution. 1-D
          array of same length as :code:`var` if :code:`covariate` is not
          :code:`None`, otherwise a single value.
        - :code:`"df_prior"`, degrees of freedom of prior distribution. 1-D
          array of same length as :code:`var` if :code:`robust=True`, otherwise
          a single value.
    """
    var = np.array(var, dtype=float, copy=True)
    n = len(var)

    # Degenerate special cases
    if n == 0:
        raise ValueError("var is empty")
    if n < 3:
        return {"var_post": var, "var_prior": var, "df_prior": 0}

    # When df==0, guard against missing or infinite values in var
    df = np.asarray(df, dtype=float)
    if df.size > 1:
        var[df == 0] = 0

    if span is not None:
        legacy = False
    if legacy is None:
        positive_df = df[df > 0]
        # R's min/max comparison also selects legacy when df contains NA;
        # fitFDist then omits those observations.
        legacy = np.isnan(df).any() or (
            positive_df.size > 0 and np.all(positive_df == positive_df[0])
        )
    df = np.broadcast_to(df, var.shape)

    # Estimate hyperparameters
    if robust:
        raise NotImplementedError("Robust estimation in squeezeVar is not implemented")
    if legacy:
        fit = fitFDist(var, df1=df, covariate=covariate)
    else:
        fit = fitFDistUnequalDF1(var, df1=df, covariate=covariate, span=span)
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
