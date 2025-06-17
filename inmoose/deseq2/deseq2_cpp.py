# -----------------------------------------------------------------------------
# Copyright (C) 2013-2022 Michael I. Love, Constantin Ahlmann-Eltze
# Copyright (C) 2023-2024 Maximilien Colange

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

# DESeq2 C++ functions ported back to Python.
# DESeq2 C++ functions use Armadillo for linear algebra operations (matrices
# and vectors indexing and slicing, matrices and vectors multiplication...)
# These capabilities are in fact performed in C directly by numpy, so we
# figure that there is no need to use Armadillo here, nor to remain full C++.
# This may change in the future, e.g. for performance reasons.
#
# Note: the canonical, up-to-date DESeq2.cpp lives in the DESeq2 library, the
# development branch of which can be viewed here:
# https://github.com/mikelove/DESeq2/blob/master/src/DESeq2.cpp

import numpy as np
import pandas as pd
from scipy.special import digamma, polygamma, xlog1py, xlogy
from scipy.special import loggamma as lgamma

from ..utils import dnbinom_mu


def log_posterior(
    log_alpha,
    y,
    mu,
    x,
    log_alpha_prior_mean,
    log_alpha_prior_sigmasq,
    usePrior,
    weights,
    useWeights,
    weightThreshold,
    useCR,
):
    """
    This function returns the log posterior of dispersion parameter alpha, for
    negative binomial variables.  Given the counts :code:`y`, the expected
    means :code:`mu`, the design matrix :code:`x` (used for calculating the
    Cox-Reid adjustment), and the parameters for the normal prior on
    :code:`log_alpha`.

    Arguments
    ---------
    log_alpha : ndarray
        log of the dispersion parameters alpha, last dim is M (or broadcastable)
    y : ndarray
        counts, matrix of shape (N,M) or vector of length N
    mu : ndarray
        expected means, same shape as :code:`y`
    x : ndarray
        design matrix, shape (N,P)
    log_alpha_prior_mean : ndarray
        normal prior on log alpha, vector of length M
    log_alpha_prior_sigmasq : float
        standard deviation of the prior on log alpha
    usePrior : bool
        whether to use prior regularization
    weights : ndarray
        weights to apply, same shape as :code:`y`
    useWeights : bool
        whether to use weights
    weightThreshold : float
        minimal weight to consider in Cox-Reid regularization
    useCR : bool
        whether to use Cox-Reid regularization

    Returns
    -------
    ndarray
        the log value of the posterior, weighted and regularized as specified
        by the input arguments
        same shape as first argument :code:`log_alpha`
    """

    # make sure that all arrays are broadcastable to the appropriate shapes
    if not isinstance(log_alpha, np.ndarray):
        log_alpha = np.repeat(log_alpha, 1)
    if len(y.shape) == 1:
        y = y[:, None]
    if len(mu.shape) == 1:
        mu = mu[:, None]
    if len(weights.shape) == 1:
        weights = weights[:, None]
    # helper variables to control the shapes
    M = np.maximum(y.shape[-1], log_alpha.shape[-1])
    (N, P) = x.shape

    alpha = np.exp(log_alpha)
    if useCR:
        x = x[None]
        # NB: now, x.shape == (1, N, P)
        mu_neg1 = 1.0 / mu
        w_diag = 1.0 / (mu_neg1 + np.expand_dims(alpha, axis=-2))
        if useWeights:
            # cancel out all weights below the threshold
            idx = weights <= weightThreshold
            w_diag[np.broadcast_to(idx, w_diag.shape)] = 0.0
        assert w_diag.shape[-2:] == (N, M)
        assert w_diag.shape[:-2] == log_alpha.shape[:-1]

        # use `np.swapaxes` to transpose the matrices stored in the last 2 dims
        w_diag = np.swapaxes(w_diag, -1, -2)
        # insert a new axis in last position
        w_diag = np.expand_dims(w_diag, axis=-1)
        assert w_diag.shape[-3:] == (M, N, 1)
        assert w_diag.shape[:-3] == log_alpha.shape[:-1]
        b = np.swapaxes(x * w_diag, -1, -2) @ x
        assert b.shape[-3:] == (M, P, P)
        assert b.shape[:-3] == log_alpha.shape[:-1]
        cr_term = -0.5 * np.linalg.slogdet(b)[1]
        assert cr_term.shape[:-1] == log_alpha.shape[:-1]
        assert cr_term.shape[-1] == M
    else:
        cr_term = 0.0

    # insert a new axis before the last one, to broadcast on the N dimension
    # of y, mu, weights
    alpha = np.expand_dims(alpha, axis=-2)
    alpha_neg1 = 1.0 / alpha
    if useWeights:
        ll_part = np.sum(
            weights
            * (
                lgamma(y + alpha_neg1)
                - lgamma(alpha_neg1)
                - xlogy(y, mu + alpha_neg1)
                - xlog1py(alpha_neg1, alpha * mu)
            ),
            axis=-2,
        )
    else:
        ll_part = np.sum(
            lgamma(y + alpha_neg1)
            - lgamma(alpha_neg1)
            - xlogy(y, mu + alpha_neg1)
            - xlog1py(alpha_neg1, alpha * mu),
            axis=-2,
        )

    assert ll_part.shape[:-1] == log_alpha.shape[:-1], (
        f"{ll_part.shape} vs {log_alpha.shape}"
    )
    assert ll_part.shape[-1] == M

    if usePrior:
        prior_part = (
            -0.5 * (log_alpha - log_alpha_prior_mean) ** 2 / log_alpha_prior_sigmasq
        )
        assert prior_part.shape[:-1] == log_alpha.shape[:-1], (
            f"{prior_part.shape} vs {log_alpha.shape}"
        )
    else:
        prior_part = 0.0

    return ll_part + prior_part + cr_term


def dlog_posterior(
    log_alpha,
    y,
    mu,
    x,
    log_alpha_prior_mean,
    log_alpha_prior_sigmasq,
    usePrior,
    weights,
    useWeights,
    weightThreshold,
    useCR,
):
    """
    This function returns the derivative of the log posterior with respect to
    the log of the dispersion parameter alpha, given the same inputs as
    :func:`log_posterior`.

    Arguments
    ---------
    log_alpha : ndarray
        log of the dispersion parameters alpha, last dim is M (or broadcastable)
    y : ndarray
        counts, matrix of shape (N,M) or vector of length N
    mu : ndarray
        expected means, same shape as :code:`y`
    x : ndarray
        design matrix, shape (N,P)
    log_alpha_prior_mean : ndarray
        normal prior on log alpha, vector of length M
    log_alpha_prior_sigmasq : float
        standard deviation of the prior on log alpha
    usePrior : bool
        whether to use prior regularization
    weights : ndarray
        weights to apply, same shape as :code:`y`
    useWeights : bool
        whether to use weights
    weightThreshold : float
        minimal weight to consider in Cox-Reid regularization
    useCR : bool
        whether to use Cox-Reid regularization

    Returns
    -------
    ndarray
        the derivative of the log value of the posterior, weighted and
        regularized as specified by the input arguments
        same shape as first argument :code:`log_alpha`
    """

    # make sure that all arrays are broadcastable to the appropriate shapes
    if not isinstance(log_alpha, np.ndarray):
        log_alpha = np.repeat(log_alpha, 1)
    if len(y.shape) == 1:
        y = y[:, None]
    if len(mu.shape) == 1:
        mu = mu[:, None]
    if len(weights.shape) == 1:
        weights = weights[:, None]

    # helper variables to control the shapes
    M = log_alpha.shape[0]
    (N, P) = x.shape

    alpha = np.exp(log_alpha)
    if useCR:
        x = x[None]
        # NB: now, x.shape == (1, N, P)
        mu_neg1 = 1.0 / mu
        w_diag = 1.0 / (mu_neg1 + alpha[None])
        dw_diag = -np.power(mu_neg1 + alpha[None], -2)
        assert w_diag.shape == (N, M)
        assert dw_diag.shape == (N, M)
        # NB: w_diag.shape == dw_diag.shape == mu.shape == (N, M)
        if useWeights:
            # cancel out all weights below the threshold
            idx = weights <= weightThreshold
            w_diag[np.broadcast_to(idx, w_diag.shape)] = 0.0
            dw_diag[np.broadcast_to(idx, dw_diag.shape)] = 0.0

        # use `np.swapaxes` to transpose the matrices stored in the last 2 dims
        w_diag = np.swapaxes(w_diag, -1, -2)
        dw_diag = np.swapaxes(dw_diag, -1, -2)
        # insert a new axis in last position
        w_diag = np.expand_dims(w_diag, axis=-1)
        dw_diag = np.expand_dims(dw_diag, axis=-1)
        b = np.swapaxes(x * w_diag, -1, -2) @ x
        db = np.swapaxes(x * dw_diag, -1, -2) @ x
        assert b.shape == (M, P, P)
        assert db.shape == (M, P, P)
        cr_term = -0.5 * np.trace(np.linalg.inv(b) @ db, axis1=-2, axis2=-1)
        # NB original code computes
        #   ddetb = det(b) * trace(b.i() * db)
        # then
        #   cr_term = -0.5 * ddetb / det(b)
        # not sure why they multiply/divide by det(b)...
        assert cr_term.shape == alpha.shape, f"{cr_term.shape} vs {alpha.shape}"
    else:
        cr_term = 0.0

    alpha = alpha[None]
    alpha_neg1 = 1.0 / alpha
    alpha_neg2 = np.power(alpha, -2)
    alphamu = alpha * mu
    if useWeights:
        ll_part = alpha_neg2.squeeze() * np.sum(
            weights
            * (
                digamma(alpha_neg1)
                + np.log(1 + alphamu)
                - alphamu / (1.0 + alphamu)
                - digamma(y + alpha_neg1)
                + y / (mu + alpha_neg1)
            ),
            axis=0,
        )
    else:
        ll_part = alpha_neg2.squeeze() * np.sum(
            digamma(alpha_neg1)
            + np.log(1 + alphamu)
            - alphamu / (1.0 + alphamu)
            - digamma(y + alpha_neg1)
            + y / (mu + alpha_neg1),
            axis=0,
        )

    # only the prior part is wrt log alpha
    if usePrior:
        prior_part = -1.0 * (log_alpha - log_alpha_prior_mean) / log_alpha_prior_sigmasq
    else:
        prior_part = 0.0

    # note: return dlog_post / dalpha * alpha because we take derivatives wrt log alpha
    return (ll_part + cr_term) * alpha.squeeze() + prior_part


def d2log_posterior(
    log_alpha,
    y,
    mu,
    x,
    log_alpha_prior_mean,
    log_alpha_prior_sigmasq,
    usePrior,
    weights,
    useWeights,
    weightThreshold,
    useCR,
):
    """
    This function returns the second derivative of the log posterior with
    respect to the log of the dispersion parameter alpha, given the same inputs
    as :func:`log_posterior`.

    Arguments
    ---------
    log_alpha : ndarray
        log of the dispersion parameters alpha, last dim is M (or broadcastable)
    y : ndarray
        counts, matrix of shape (N,M) or vector of length N
    mu : ndarray
        expected means, same shape as :code:`y`
    x : ndarray
        design matrix, shape (N,P)
    log_alpha_prior_mean : ndarray
        normal prior on log alpha, vector of length M
    log_alpha_prior_sigmasq : float
        standard deviation of the prior on log alpha
    usePrior : bool
        whether to use prior regularization
    weights : ndarray
        weights to apply, same shape as :code:`y`
    useWeights : bool
        whether to use weights
    weightThreshold : float
        minimal weight to consider in Cox-Reid regularization
    useCR : bool
        whether to use Cox-Reid regularization

    Returns
    -------
    ndarray
        the second derivative of the log value of the posterior, weighted and
        regularized as specified by the input arguments
        same shape as first argument :code:`log_alpha`
    """

    # make sure that all arrays are broadcastable to the appropriate shapes
    if not isinstance(log_alpha, np.ndarray):
        log_alpha = np.repeat(log_alpha, 1)
    if len(y.shape) == 1:
        y = y[:, None]
    if len(mu.shape) == 1:
        mu = mu[:, None]
    if len(weights.shape) == 1:
        weights = weights[:, None]

    # helper variables to control the shapes
    M = log_alpha.shape[0]
    (N, P) = x.shape

    alpha = np.exp(log_alpha)
    if useCR:
        x = x[None]
        # NB: now, x.shape == (1, N, P)
        mu_neg1 = 1.0 / mu
        w_diag = 1.0 / (mu_neg1 + alpha[None])
        dw_diag = -1.0 * np.power(mu_neg1 + alpha[None], -2)
        d2w_diag = 2.0 * np.power(mu_neg1 + alpha[None], -3)
        assert w_diag.shape == (N, M)
        assert dw_diag.shape == (N, M)
        assert d2w_diag.shape == (N, M)
        # NB: w_diag.shape == dw_diag.shape == d2w_diag.shape == mu.shape == (N, M)
        if useWeights:
            # cancel out all weights below the threshold
            idx = weights <= weightThreshold
            w_diag[np.broadcast_to(idx, w_diag.shape)] = 0.0
            dw_diag[np.broadcast_to(idx, dw_diag.shape)] = 0.0
            d2w_diag[np.broadcast_to(idx, d2w_diag.shape)] = 0.0

        # use `np.swapaxes` to transpose the matrices stored in the last 2 dims
        w_diag = np.swapaxes(w_diag, -1, -2)
        dw_diag = np.swapaxes(dw_diag, -1, -2)
        d2w_diag = np.swapaxes(d2w_diag, -1, -2)
        # insert a new axis in last position
        w_diag = np.expand_dims(w_diag, axis=-1)
        dw_diag = np.expand_dims(dw_diag, axis=-1)
        d2w_diag = np.expand_dims(d2w_diag, axis=-1)
        b = np.swapaxes(x * w_diag, -1, -2) @ x
        b_i = np.linalg.inv(b)
        db = np.swapaxes(x * dw_diag, -1, -2) @ x
        d2b = np.swapaxes(x * d2w_diag, -1, -2) @ x
        assert b.shape == (M, P, P)
        assert db.shape == (M, P, P)
        assert d2b.shape == (M, P, P)

        ddetb = np.trace(b_i @ db, axis1=-2, axis2=-1)
        d2detb = (
            np.power(ddetb, 2)
            - np.trace(b_i @ db @ b_i @ db, axis1=-2, axis2=-1)
            + np.trace(b_i @ d2b, axis1=-2, axis2=-1)
        )
        cr_term = 0.5 * np.power(ddetb, 2) - 0.5 * d2detb
        assert cr_term.shape == alpha.shape, f"{cr_term.shape} vs {alpha.shape}"
        x = x.squeeze(-3)
    else:
        cr_term = 0.0

    alpha = alpha[None]
    alpha_neg1 = 1.0 / alpha
    alpha_neg2 = np.power(alpha, -2)
    alphamu = alpha * mu
    if useWeights:
        ll_part = -2 * np.power(alpha, -3) * np.sum(
            weights
            * (
                digamma(alpha_neg1)
                + np.log(1 + alphamu)
                - alphamu / (1 + alphamu)
                - digamma(y + alpha_neg1)
                + y / (mu + alpha_neg1)
            )
        ) + alpha_neg2 * np.sum(
            weights
            * (
                -1 * alpha_neg2 * polygamma(1, alpha_neg1)
                + np.power(mu, 2) * alpha * np.power(1 + alphamu, -2)
                + alpha_neg2 * polygamma(1, y + alpha_neg1)
                + alpha_neg2 * y * np.power(mu + alpha_neg1, -2)
            )
        )
    else:
        ll_part = -2 * np.power(alpha, -3) * np.sum(
            digamma(alpha_neg1)
            + np.log(1 + alphamu)
            - alphamu / (1 + alphamu)
            - digamma(y + alpha_neg1)
            + y / (mu + alpha_neg1)
        ) + alpha_neg2 * np.sum(
            -1 * alpha_neg2 * polygamma(1, alpha_neg1)
            + np.power(mu, 2) * alpha * np.power(1 + alphamu, -2)
            + alpha_neg2 * polygamma(1, y + alpha_neg1)
            + alpha_neg2 * y * np.power(mu + alpha_neg1, -2)
        )

    # only the prior part is wrt log alpha
    if usePrior:
        prior_part = -1.0 / log_alpha_prior_sigmasq
    else:
        prior_part = 0.0

    # note: return (d2log_post/dalpha2 * alpha^2 + dlog_post/dalpha * alpha)
    #           =  (d2log_post/dalpha2 * alpha^2 + dlog_post/dlogalpha)
    # because we take derivatives wrt log alpha
    res = (
        (ll_part + cr_term) * np.power(alpha, 2)
        + dlog_posterior(
            log_alpha,
            y,
            mu,
            x,
            log_alpha_prior_mean,
            log_alpha_prior_sigmasq,
            False,
            weights,
            useWeights,
            weightThreshold,
            useCR,
        )
    ) + prior_part
    return res


def fitDisp(
    y,
    x,
    mu_hat,
    log_alpha,
    log_alpha_prior_mean,
    log_alpha_prior_sigmasq,
    min_log_alpha,
    kappa_0,
    tol,
    maxit,
    usePrior,
    weights,
    useWeights,
    weightThreshold,
    useCR,
):
    """
    Fit dispersions for negative binomial GLMs.

    This function estimates the dispersion parameter (alpha) for negative
    binomial generalized linear models. The fitting is performed on the log
    scale.

    Arguments
    ---------
    y : ndarray
        matrix of counts, shape (N,M)
    x : ndarray
        design matrix, shape (M,K)
    mu_hat : ndarray
        the expected mean values, shape (N,M)
    log_alpha : ndarray
        vector of initial guesses for log(alpha), shape N
    log_alpha_prior_mean : ndarray
        vector of fitted values for log(alpha), shape N
    log_alpha_prior_sigmasq : float
        the variance of the prior
    min_log_alpha : float
        the minimum value for log(alpha)
    kappa_0 : float
        parameter for initial proposal in the backtracking search.
        initial proposal = log(alpha) + kappa_0 * d(log-likelihood)/d(log(alpha))
    tol : float
        tolerance for convergence estimates
    maxit : int
        maximum number of iterations
    usePrior : bool
        whether to use a priori or just compute the MLE
    weights : ndarray
        observation weights, shape (N,M)
    useWeights : bool
        whether to use weights
    weightThreshold : float
        the threshold for subsetting the design matrix and GLM weights to
        calculate the Cox-Reid correction
    useCR : bool
        whether to use the Cox-Reid correction

    Returns
    -------
    log_alpha : ndarray
        the fitted dispersion parameters, on the log scale. Shape N.
    iter : ndarray
        the number of iterations for each parameter. Shape N.
    iter_accept : ndarray
        the number of accepted proposals for each parameter. Shape N.
    last_change : ndarray
        the last change of the fitted dispersion parameters. Shape N.
    initial_lp : ndarray
        the initial log posterior values. Shape N.
    initial_dlp : ndarray
        the initial derivatives (wrt. log(alpha)) of the log posterior. Shape N.
    last_lp : ndarray
        the last log posterior values. Shape N.
    last_dlp : ndarray
        the last derivatives (wrt. log(alpha)) of the log posterior. Shape N.
    last_d2lp : ndarray
        the last second derivatives (wrt. log(alpha)) of the log posterior. Shape N.
    """

    if isinstance(log_alpha, pd.Series):
        log_alpha = log_alpha.values
    if isinstance(log_alpha, (int, float)):
        log_alpha = np.repeat(float(log_alpha), y.shape[1])
    if isinstance(log_alpha_prior_mean, pd.Series):
        log_alpha_prior_mean = log_alpha_prior_mean.values
    if isinstance(log_alpha_prior_mean, (int, float)):
        log_alpha_prior_mean = np.repeat(float(log_alpha_prior_mean), y.shape[1])
    assert y.shape[1] == mu_hat.shape[1]
    assert y.shape[1] == log_alpha.shape[0]
    assert y.shape[1] == log_alpha_prior_mean.shape[0]
    assert y.shape == weights.shape

    y_n = y.shape[1]
    epsilon = 1.0e-4
    iter_ = np.zeros(y_n)
    iter_accept = np.zeros(y_n)

    # maximize the log likelihood over the variable a, the log of alpha, the
    # dispersion parameter.
    # in order to express the optimization in a typical manner, for calculating
    # theta(kappa) we multiply the log likelihood by -1 and seek a minimum
    # we use a line search based on the Armijo rule.
    # define a function theta(kappa) = f(a + kappa * d) where d is the search
    # direction
    # in this case the search direction is given by the first derivative of the
    # log likelihood
    a = log_alpha.copy()
    lp = log_posterior(
        a,
        y,
        mu_hat,
        x,
        log_alpha_prior_mean,
        log_alpha_prior_sigmasq,
        usePrior,
        weights,
        useWeights,
        weightThreshold,
        useCR,
    )
    dlp = dlog_posterior(
        a,
        y,
        mu_hat,
        x,
        log_alpha_prior_mean,
        log_alpha_prior_sigmasq,
        usePrior,
        weights,
        useWeights,
        weightThreshold,
        useCR,
    )

    kappa = np.repeat(kappa_0, y_n)
    initial_lp = lp.copy()
    initial_dlp = dlp.copy()
    change = np.repeat(-1.0, y_n)
    last_change = np.repeat(-1.0, y_n)

    idx = np.repeat(True, y_n)
    for t in range(maxit):
        # iter_ counts the number of steps taken out of maxit
        iter_[idx] += 1
        a_propose = a + kappa * dlp
        # note: lgamma is unstable for values around 1e17, where there is a
        # switch in lgamma.c
        # we limit log alpha from going lower than -30
        kappa = np.where(a_propose < -30.0, (-30.0 - a) / dlp, kappa)
        # we limit log alpha from going higher than 10
        kappa = np.where(a_propose > 10.0, (10.0 - a) / dlp, kappa)

        lpost = log_posterior(
            a[idx] + kappa[idx] * dlp[idx],
            y[:, idx],
            mu_hat[:, idx],
            x,
            log_alpha_prior_mean[idx],
            log_alpha_prior_sigmasq,
            usePrior,
            weights[:, idx],
            useWeights,
            weightThreshold,
            useCR,
        )
        theta_kappa = np.zeros(y_n)
        theta_kappa[idx] = -lpost
        theta_hat_kappa = -lp - kappa * epsilon * np.power(dlp, 2)

        # if this inequality is true, we have satisfied the Armijo rule and
        # accept the step size kappa, otherwise we halve kappa
        armijo_idx = idx & (theta_kappa <= theta_hat_kappa)
        # iter_accept counts the number of accepted proposals
        iter_accept[armijo_idx] += 1
        a[armijo_idx] = (a + kappa * dlp)[armijo_idx]
        lpnew = np.zeros(y_n)
        lpnew[idx] = lpost
        # look for change in log likelihood
        change[armijo_idx] = lpnew[armijo_idx] - lp[armijo_idx]
        lp = np.where(armijo_idx & (change < tol), lpnew, lp)
        idx[armijo_idx & (change < tol)] = False
        # if log(alpha) is going to -infinity, break the loop
        idx[armijo_idx & (a < min_log_alpha)] = False

        idx2 = idx & armijo_idx
        lp[idx2] = lpnew[idx2]
        dlp[idx2] = dlog_posterior(
            a[idx2],
            y[:, idx2],
            mu_hat[:, idx2],
            x,
            log_alpha_prior_mean[idx2],
            log_alpha_prior_sigmasq,
            usePrior,
            weights[:, idx2],
            useWeights,
            weightThreshold,
            useCR,
        )
        # instead of resetting kappa to kappa_0
        # multiply kappa by 1.1
        kappa[idx2] = np.minimum(kappa[idx2] * 1.1, kappa_0)
        # every 5 accepts, halve kappa
        # to prevent slow convergence due to overshooting
        kappa[idx2 & (iter_accept % 5 == 0)] /= 2.0

        kappa[~armijo_idx] /= 2.0

    last_lp = lp
    last_dlp = dlp
    last_d2lp = d2log_posterior(
        a,
        y,
        mu_hat,
        x,
        log_alpha_prior_mean,
        log_alpha_prior_sigmasq,
        usePrior,
        weights,
        useWeights,
        weightThreshold,
        useCR,
    )
    log_alpha = a
    # last change indicates the change for the final iteration
    last_change = change

    return {
        "log_alpha": log_alpha,
        "iter": iter_,
        "iter_accept": iter_accept,
        "last_change": last_change,
        "initial_lp": initial_lp,
        "initial_dlp": initial_dlp,
        "last_lp": last_lp,
        "last_dlp": last_dlp,
        "last_d2lp": last_d2lp,
    }


def fitDispWrapper(**kwargs):
    """
    Wrapper around :func:`fitDisp` to check for NaN in arguments

    See also
    --------
    fitDisp
    """
    for k, v in kwargs.items():
        if np.any(np.isnan(v)):
            raise ValueError(f"argument {k} of fitDisp contains a NaN value")
    return fitDisp(**kwargs)


def fitDispGrid(
    y,
    x,
    mu_hat,
    disp_grid,
    log_alpha_prior_mean,
    log_alpha_prior_sigmasq,
    usePrior,
    weights,
    useWeights,
    weightThreshold,
    useCR,
):
    """
    Fit dispersions by evaluating over a grid

    This function estimates the dispersion parameters (alpha) for negative
    binomial generalized linear models. The fitting is performed on the log
    scale.

    Arguments
    ---------
    y : ndarray
        matrix of counts, shape (N,M)
    x : ndarray
        design matrix, shape (M,K)
    mu_hat : ndarray
        the expected mean values, shape (N,M)
    disp_grid : ndarray
        the grid over which to estimate
    log_alpha_prior_mean : ndarray
        vector of fitted values for log(alpha), shape N
    log_alpha_prior_sigmasq : float
        the variance of the prior
    usePrior : bool
        whether to use a priori or just compute the MLE
    weights : ndarray
        observation weights, shape (N,M)
    useWeights : bool
        whether to use weights
    weightThreshold : float
        the threshold for subsetting the design matrix and GLM weights to
        calculate the Cox-Reid correction
    useCR : bool
        whether to use the Cox-Reid correction

    Returns
    -------
    ndarray
        the estimated dispersion parameters, on the log scale. Shape N.
    """

    y_n = y.shape[1]
    disp_grid_n = disp_grid.shape[0]
    delta = disp_grid[1] - disp_grid[0]

    logpostvec = log_posterior(
        disp_grid[:, None],
        y,
        mu_hat,
        x,
        log_alpha_prior_mean,
        log_alpha_prior_sigmasq,
        usePrior,
        weights,
        useWeights,
        weightThreshold,
        useCR,
    )
    idxmax = np.argmax(logpostvec, axis=0)
    assert idxmax.shape == (y_n,)
    a_hat = disp_grid[idxmax]
    disp_grid_fine = np.linspace(a_hat - delta, a_hat + delta, disp_grid_n)
    assert disp_grid_fine.shape == (disp_grid_n, y_n)
    logpostvec = log_posterior(
        disp_grid_fine,
        y,
        mu_hat,
        x,
        log_alpha_prior_mean,
        log_alpha_prior_sigmasq,
        usePrior,
        weights,
        useWeights,
        weightThreshold,
        useCR,
    )
    idxmax = np.argmax(logpostvec, axis=0)
    assert idxmax.shape == (y_n,)
    log_alpha = np.take_along_axis(disp_grid_fine, idxmax[None], axis=0).squeeze(0)
    assert log_alpha.shape == (y_n,)
    return log_alpha


def fitDispGridWrapper(**kwargs):
    """
    Wrapper around :func:`fitDispGrid`

    This wrapper checks for NaN in arguments, and automatically builds a grid
    on which :func:`fitDispGrid` will be called. Contrary to
    :func:`fitDispGrid`, it returns the estimated dispersion parameters on the
    natural scale.

    See also
    --------
    fitDispGrid

    Returns
    -------
    ndarray
        the estimated dispersion parameters, on the natural scale. Shape N.
    """
    for k, v in kwargs.items():
        if np.any(np.isnan(v)):
            raise ValueError(f"argument {k} of fitDispGrid contains a NaN value")

    minLogAlpha = np.log(1e-8)
    maxLogAlpha = np.log(np.maximum(10, kwargs["y"].shape[0]))
    dispGrid = np.linspace(minLogAlpha, maxLogAlpha, 20)
    kwargs["mu_hat"] = kwargs["mu"]
    # Convert pandas Series to numpy array if needed
    if isinstance(kwargs["log_alpha_prior_mean"], pd.Series):
        kwargs["log_alpha_prior_mean"] = kwargs["log_alpha_prior_mean"].values
    del kwargs["mu"]
    kwargs["disp_grid"] = dispGrid
    logAlpha = fitDispGrid(**kwargs)
    return np.exp(logAlpha)


def fitBeta(
    y,
    x,
    nf,
    alpha_hat,
    contrast,
    beta_mat,
    lambda_,
    weights,
    useWeights,
    tol,
    maxit,
    useQR,
    minmu,
):
    """
    Fit beta coefficients for negative binomial GLMs

    This function estimates the coefficients (beta) for negative binomial
    generalized linear models. Fitting is performed on the log scale.

    Arguments
    ---------
    y : ndarray
        matrix of counts, shape (N,M)
    x : ndarray
        design matrix, shape (M,K)
    nf : ndarray
        matrix of normalization factors, shape (N,M)
    alpha_hat : ndarray
        vector of the dispersion estimates, shape N
    contrast : array-like
        vector for a possible contrast, shape K
    beta_mat : ndarray
        the initial estimates for the betas, shape (N,K)
    lambda_ : ndarray
        the ridge values, shape K
    weights : ndarray
        observation weights, shape (N,M)
    useWeights : bool
        whether to use weights
    tol : float
        tolerance for convergence estimates
    maxit : int
        maximum number of iterations
    useQR : bool
        whether to use QR decomposition

    Returns
    -------
    beta_mat : ndarray
        the fitted coefficients, on the log scale. Shape (N,K)
    beta_var_mat : ndarray
        the variance of the fitted coefficients. Shape (N,K)
    iter : ndarray
        the number of iterations for each row. Shape N
    hat_diagonals : ndarray
        TODO
    contrast_num : ndarray
        TODO
    contrast_denom : ndarray
        TODO
    deviance : ndarray
        TODO
    """
    y_m, y_n = y.shape
    x_p = x.shape[1]

    assert not isinstance(alpha_hat, pd.Series)
    assert alpha_hat.shape == (y_n,)
    assert beta_mat.shape == (y_n, x_p)
    assert y.shape[0] == x.shape[0]
    assert nf.shape == y.shape
    assert weights.shape == y.shape
    assert lambda_.ndim == 1
    assert lambda_.shape[0] == x.shape[1], f"{lambda_.shape}, {x.shape}"

    beta_var_mat = np.zeros(beta_mat.shape)
    contrast_num = np.zeros(beta_mat.shape[0])
    contrast_denom = np.zeros(beta_mat.shape[0])
    # bound the estimated count, as weights include 1/mu
    large = 30.0
    iter_ = np.zeros(y_n)
    ridge = np.diag(lambda_)

    mu_hat = np.maximum(nf * np.exp(x @ beta_mat.T), minmu)
    assert mu_hat.shape == nf.shape
    dev = np.zeros(y_n)
    dev_old = np.zeros(y_n)
    idx = np.repeat(True, y_n)

    for t in range(maxit):
        idx_n = np.sum(idx)
        iter_[idx] += 1

        mu_hat_idx = mu_hat[:, idx]
        if useWeights:
            w_vec = weights[:, idx] * mu_hat_idx / (1.0 + alpha_hat[idx] * mu_hat_idx)
        else:
            w_vec = mu_hat_idx / (1.0 + alpha_hat[idx] * mu_hat_idx)
        assert w_vec.shape == (y_m, idx_n)
        w_sqrt_vec = np.sqrt(w_vec)

        if useQR:
            # make an orthonormal design matrix including the ridge penalty
            # prepare matrices
            weighted_x_ridge = np.hstack(
                [
                    x * w_sqrt_vec.T[:, :, None],
                    np.broadcast_to(np.sqrt(ridge), (idx_n, x_p, x_p)),
                ]
            )
            assert weighted_x_ridge.shape == (idx_n, y_m + x_p, x_p)
            q, r = np.linalg.qr(weighted_x_ridge)
            assert q.shape[0] == r.shape[0] == idx_n
            big_w_diag = np.ones((idx_n, y_m + x_p))
            big_w_diag[:, :y_m] = w_vec.T
            z = np.log(mu_hat_idx / nf[:, idx]) + (y[:, idx] - mu_hat_idx) / mu_hat_idx
            assert z.shape == (y_m, idx_n)
            w_diag = w_vec.copy()
            z_sqrt_w = z * np.sqrt(w_diag)
            big_z_sqrt_w = np.zeros((idx_n, y_m + x_p))
            big_z_sqrt_w[:, :y_m] = z_sqrt_w.T
            # IRLS with Q matrix for X
            gamma_hat = np.swapaxes(q, -1, -2) @ big_z_sqrt_w[:, :, None]
            beta_hat = np.linalg.solve(r, gamma_hat).squeeze(-1)
        else:
            # use the standard design matrix and matrix inversion
            z = np.log(mu_hat_idx / nf[:, idx]) + (y[:, idx] - mu_hat_idx) / mu_hat_idx
            assert (x.T @ (x * w_vec.T[:, :, None]) + ridge).shape == (idx_n, x_p, x_p)
            zwtx = (z * w_vec).T @ x
            assert (zwtx).shape == (idx_n, x_p)
            beta_hat = np.linalg.solve(
                x.T @ (x * w_vec.T[:, :, None]) + ridge, zwtx[:, :, None]
            ).squeeze(-1)

        assert beta_hat.shape == (idx_n, x_p), f"{beta_hat.shape} vs {(idx_n, x_p)}"
        # easier to update beta_mat here instead than at the end of the loop
        beta_mat[idx, :] = beta_hat
        subidx = ~(np.sum(np.abs(beta_hat) > large, axis=1) > 0)
        assert subidx.shape == (idx_n,)
        newidx = idx.copy()
        newidx[idx] = subidx
        iter_[idx & ~newidx] = maxit

        idx = newidx
        mu_hat[:, idx] = np.maximum(
            nf[:, idx] * np.exp(x @ beta_hat[subidx, :].T), minmu
        )
        dev[idx] = 0.0
        if useWeights:
            dev[idx] -= 2.0 * np.sum(
                weights[:, idx]
                * dnbinom_mu(y[:, idx], 1.0 / alpha_hat[idx], mu_hat[:, idx], True),
                axis=0,
            )
        else:
            dev[idx] -= 2.0 * np.sum(
                dnbinom_mu(y[:, idx], 1.0 / alpha_hat[idx], mu_hat[:, idx], True),
                axis=0,
            )

        conv_test = np.abs(dev - dev_old) / (np.abs(dev) + 0.1)
        nanidx = np.isnan(conv_test)
        iter_[idx & nanidx] = maxit
        idx &= ~nanidx
        if t > 0:
            idx &= ~(conv_test < tol)
        dev_old = dev.copy()

    # recalculate w so that this is identical if we start with beta_hat
    if useWeights:
        w_vec = weights * mu_hat / (1.0 + alpha_hat * mu_hat)
    else:
        w_vec = mu_hat / (1.0 + alpha_hat * mu_hat)
    w_sqrt_vec = np.sqrt(w_vec)

    xw = x * w_sqrt_vec.T[:, :, None]
    assert xw.shape == (y_n, y_m, x_p)
    xtwxr_inv = np.linalg.inv(x.T @ (x * w_vec.T[:, :, None]) + ridge)
    assert xtwxr_inv.shape == (y_n, x_p, x_p)

    hat_diagonals = np.zeros(y.shape)
    # this is equivalent to (for all j):
    #   hat_diagonals[:,j] = np.diag(xw[j] @ xtwxr_inv[j] @ xw[j].T)
    # but it avoids computing full matrix products just to retrieve the diags
    for k in range(x_p):
        for m in range(x_p):
            hat_diagonals[:, :] += (xw[:, :, k] * xw[:, :, m]).T * xtwxr_inv[:, k, m]

    # sigma is the covariance matrix for the betas
    sigma = xtwxr_inv @ x.T @ (x * w_vec.T[:, :, None]) @ xtwxr_inv
    assert sigma.shape == (y_n, x_p, x_p)
    contrast_num = contrast @ beta_mat.T
    contrast_denom = np.sqrt(contrast.T @ sigma @ contrast)
    beta_var_mat = np.diagonal(sigma, axis1=-2, axis2=-1)

    return {
        "beta_mat": beta_mat,
        "beta_var_mat": beta_var_mat,
        "iter": iter_,
        "hat_diagonals": hat_diagonals,
        "contrast_num": contrast_num,
        "contrast_denom": contrast_denom,
        "deviance": dev,
    }


def fitBetaWrapper(**kwargs):
    """
    Wrapper around :func:`fitBeta`

    This wrapper checks for NaN in arguments. It also sets a default contrast
    if none is provided.

    See also
    --------
    fitBeta
    """
    for k, v in kwargs.items():
        if np.any(np.isnan(v)):
            raise ValueError(f"argument {k} of fitBeta contains a NaN value")

    if "contrast" not in kwargs:
        kwargs["contrast"] = np.zeros(kwargs["x"].shape[1])
        kwargs["contrast"][0] = 1

    return fitBeta(**kwargs)
