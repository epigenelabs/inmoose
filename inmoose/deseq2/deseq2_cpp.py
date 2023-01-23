# -----------------------------------------------------------------------------
# Copyright (C) ??-2022 Michael I. Love, Constantin Ahlmann-Eltze
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
from scipy.special import loggamma as lgamma
from scipy.special import digamma, polygamma

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
    This function returns the log posterior of dispersion parameter alpha, for negative binomial variables.
    Given the counts y, the expected means my, the design matrix x (used for calculating the Cox-Reid adjustment),
    and the parameters for the normal prior on log alpha
    """
    alpha = np.exp(log_alpha)
    if useCR:
        w_diag = np.power(np.power(mu, -1) + alpha, -1)
        if useWeights:
            x = x[weights > weightThreshold, :]
            x = x[:, np.sum(np.abs(x), 0) > 0.0]
            w_diag = w_diag[weights > weightThreshold]
        b = x.T @ (x * w_diag[:, None])
        (sign, logdet) = np.linalg.slogdet(b)
        cr_term = -0.5 * logdet
    else:
        cr_term = 0.0

    alpha_neg1 = np.power(alpha, -1)
    if useWeights:
        ll_part = np.sum(
            weights
            * (
                lgamma(y + alpha_neg1)
                - lgamma(alpha_neg1)
                - y * np.log(mu + alpha_neg1)
                - alpha_neg1 * np.log(1.0 + alpha * mu)
            )
        )
    else:
        ll_part = np.sum(
            lgamma(y + alpha_neg1)
            - lgamma(alpha_neg1)
            - y * np.log(mu + alpha_neg1)
            - alpha_neg1 * np.log(1.0 + alpha * mu)
        )

    if usePrior:
        prior_part = (
            -0.5 * (log_alpha - log_alpha_prior_mean) ** 2 / log_alpha_prior_sigmasq
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
    this function returns the derivative of the log posterior with respect to the log of the
    dispersion parameter alpha, given the same inputs as the previous function
    """
    alpha = np.exp(log_alpha)
    if useCR:
        w_diag = np.power(np.power(mu, -1) + alpha, -1)
        dw_diag = -1.0 * np.power(np.power(mu, -1) + alpha, -2)
        if useWeights:
            x = x[weights > weightThreshold, :]
            x = x[:, np.sum(np.abs(x), 0) > 0.0]
            w_diag = w_diag[weights > weightThreshold]
            dw_diag = dw_diag[weights > weightThreshold]
        b = x.T @ (x * w_diag[:, None])
        db = x.T @ (x * dw_diag[:, None])
        # NB original code computes
        #   ddetb = det(b) * trace(b.i() * db)
        # then
        #   cr_term = -0.5 * ddetb / det(b)
        # not sure why they multiply/divide by det(b)...
        cr_term = -0.5 * np.trace(np.linalg.inv(b) @ db)
    else:
        cr_term = 0.0

    alpha_neg1 = np.power(alpha, -1)
    alpha_neg2 = np.power(alpha, -2)
    if useWeights:
        ll_part = alpha_neg2 * np.sum(
            weights
            * (
                digamma(alpha_neg1)
                + np.log(1 + alpha * mu)
                - alpha * mu * np.power(1.0 + alpha * mu, -1)
                - digamma(y + alpha_neg1)
                + y * np.power(mu + alpha_neg1, -1)
            )
        )
    else:
        ll_part = alpha_neg2 * np.sum(
            digamma(alpha_neg1)
            + np.log(1 + alpha * mu)
            - alpha * mu * np.power(1.0 + alpha * mu, -1)
            - digamma(y + alpha_neg1)
            + y * np.power(mu + alpha_neg1, -1)
        )

    # only the prior part is wrt log alpha
    if usePrior:
        prior_part = -1.0 * (log_alpha - log_alpha_prior_mean) / log_alpha_prior_sigmasq
    else:
        prior_part = 0.0

    # note: return dlog_post / dalpha * alpha because we take derivatives wrt log alpha
    return (ll_part + cr_term) * alpha + prior_part


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
    this function returns the second derivative of the log posterior with respect to the log of the
    dispersion parameter alpha, given the same inputs as the previous function
    """
    alpha = np.exp(log_alpha)
    x_orig = x.copy()
    if useCR:
        w_diag = np.power(np.power(mu, -1) + alpha, -1)
        dw_diag = -1 * np.power(np.power(mu, -1) + alpha, -2)
        d2w_diag = 2 * np.power(np.power(mu, -1) + alpha, -3)
        if useWeights:
            x = x[weights > weightThreshold, :]
            x = x[:, np.sum(np.abs(x), 0) > 0.0]
            w_diag = w_diag[weights > weightThreshold]
            dw_diag = dw_diag[weights > weightThreshold]
            d2w_diag = d2w_diag[weights > weightThreshold]

        b = x.T @ (x * w_diag[:, None])
        b_i = np.linalg.inv(b)
        db = x.T @ (x * dw_diag[:, None])
        d2b = x.T @ (x * d2w_diag[:, None])
        ddetb = np.trace(b_i @ db)
        d2detb = (
            np.power(np.trace(b_i @ db), 2)
            - np.trace(b_i @ db @ b_i @ db)
            + np.trace(b_i @ d2b)
        )
        cr_term = 0.5 * np.power(ddetb, 2) - 0.5 * d2detb
    else:
        cr_term = 0.0

    alpha_neg1 = np.power(alpha, -1)
    alpha_neg2 = np.power(alpha, -2)
    if useWeights:
        ll_part = -2 * np.power(alpha, -3) * np.sum(
            weights
            * (
                digamma(alpha_neg1)
                + np.log(1 + alpha * mu)
                - alpha * mu * np.power(1 + alpha * mu, -1)
                - digamma(y + alpha_neg1)
                + y * np.power(mu + alpha_neg1, -1)
            )
        ) + alpha_neg2 * np.sum(
            weights
            * (
                -1 * alpha_neg2 * polygamma(1, alpha_neg1)
                + np.power(mu, 2) * alpha * np.power(1 + alpha * mu, -2)
                + alpha_neg2 * polygamma(1, y + alpha_neg1)
                + alpha_neg2 * y * np.power(mu + alpha_neg1, -2)
            )
        )
    else:
        ll_part = -2 * np.power(alpha, -3) * np.sum(
            digamma(alpha_neg1)
            + np.log(1 + alpha * mu)
            - alpha * mu * np.power(1 + alpha * mu, -1)
            - digamma(y + alpha_neg1)
            + y * np.power(mu + alpha_neg1, -1)
        ) + alpha_neg2 * np.sum(
            -1 * alpha_neg2 * polygamma(1, alpha_neg1)
            + np.power(mu, 2) * alpha * np.power(1 + alpha * mu, -2)
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
            x_orig,
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
    if isinstance(log_alpha, (int, float)):
        log_alpha = np.repeat(float(log_alpha), y.shape[1])
    if isinstance(log_alpha_prior_mean, (int, float)):
        log_alpha_prior_mean = np.repeat(float(log_alpha_prior_mean), y.shape[1])
    assert y.shape[1] == mu_hat.shape[1]
    assert y.shape[1] == log_alpha.shape[0]
    assert y.shape[1] == log_alpha_prior_mean.shape[0]

    y_n = y.shape[1]
    epsilon = 1.0e-4
    # record log posterior values
    initial_lp = np.zeros(y_n)
    initial_dlp = np.zeros(y_n)
    last_lp = np.zeros(y_n)
    last_dlp = np.zeros(y_n)
    last_d2lp = np.zeros(y_n)
    last_change = np.zeros(y_n)
    iter_ = np.zeros(y_n)
    iter_accept = np.zeros(y_n)

    for i in range(y_n):
        # if i % 100 == 0:
        #    checkUserInterrupt()

        ycol = y[:, i]
        mu_hat_col = mu_hat[:, i]
        # maximize the log likelihood over the variable a, the log of alpha, the dispersion parameter.
        # in order to express the optimization in a typical manner,
        # for calculating theta(kappa) we multiple the log likelihood by -1 and seek a minimum
        a = log_alpha[i]
        # we use a line search based on the Armijo rule.
        # define a function theta(kappa) = f(a + kappa * d) where d is the search direction.
        # in this case the search direction is taken by the first derivative of the log likelihood
        lp = log_posterior(
            a,
            ycol,
            mu_hat_col,
            x,
            log_alpha_prior_mean[i],
            log_alpha_prior_sigmasq,
            usePrior,
            weights[:, i],
            useWeights,
            weightThreshold,
            useCR,
        )
        dlp = dlog_posterior(
            a,
            ycol,
            mu_hat_col,
            x,
            log_alpha_prior_mean[i],
            log_alpha_prior_sigmasq,
            usePrior,
            weights[:, i],
            useWeights,
            weightThreshold,
            useCR,
        )
        kappa = kappa_0
        initial_lp[i] = lp
        initial_dlp[i] = dlp
        change = -1.0
        last_change[i] = -1.0
        for t in range(maxit):
            # iter_ counts the number of steps taken out of maxit
            iter_[i] += 1
            a_propose = a + kappa * dlp
            # note: lgamma is unstable for values around 1e17, where there is a switch in lgamma.c
            # we limit log alpha from going lower than -30
            if a_propose < -30.0:
                kappa = (-30.0 - a) / dlp
            # we limit log alpha from going higher than 10
            if a_propose > 10.0:
                kappa = (10.0 - a) / dlp

            lpost = log_posterior(
                a + kappa * dlp,
                ycol,
                mu_hat_col,
                x,
                log_alpha_prior_mean[i],
                log_alpha_prior_sigmasq,
                usePrior,
                weights[:, i],
                useWeights,
                weightThreshold,
                useCR,
            )
            theta_kappa = -lpost
            theta_hat_kappa = -lp - kappa * epsilon * np.power(dlp, 2)
            # if this inequality is true, we have satisfied the Armijo rule and
            # accept the step size kappa, otherwise we halve kappa
            if theta_kappa <= theta_hat_kappa:
                # iter_accept counts the number of accepted proposals
                iter_accept[i] += 1
                a = a + kappa * dlp
                lpnew = lpost
                # look for change in log likelihood
                change = lpnew - lp
                if change < tol:
                    lp = lpnew
                    break
                # if log(alpha) is going to -infinity
                # break the loop
                if a < min_log_alpha:
                    break

                lp = lpnew
                dlp = dlog_posterior(
                    a,
                    ycol,
                    mu_hat_col,
                    x,
                    log_alpha_prior_mean[i],
                    log_alpha_prior_sigmasq,
                    usePrior,
                    weights[:, i],
                    useWeights,
                    weightThreshold,
                    useCR,
                )
                # instead of resetting kappa to kappa_0
                # multiply kappa by 1.1
                kappa = np.minimum(kappa * 1.1, kappa_0)
                # every 5 accepts, halve kappa
                # to prevent slow convergence due to overshooting
                if iter_accept[i] % 5 == 0:
                    kappa = kappa / 2.0

            else:
                kappa = kappa / 2.0

        last_lp[i] = lp
        last_dlp[i] = dlp
        last_d2lp[i] = d2log_posterior(
            a,
            ycol,
            mu_hat_col,
            x,
            log_alpha_prior_mean[i],
            log_alpha_prior_sigmasq,
            usePrior,
            weights[:, i],
            useWeights,
            weightThreshold,
            useCR,
        )
        log_alpha[i] = a
        # last change indicates the change for the final iteration
        last_change[i] = change

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
    TODO
    """
    y_n = y.shape[1]
    disp_grid_n = disp_grid.shape[0]
    delta = disp_grid[1] - disp_grid[0]
    logpostvec = np.zeros(disp_grid_n)
    log_alpha = np.zeros(y_n)

    for i in range(y_n):
        # if i % 100 == 0:
        #    checkUserInterrupt()

        ycol = y[:, i]
        mu_hat_col = mu_hat[:, i]
        for t in range(disp_grid_n):
            # maximize the log likelihood over the variable a, the log of alpha, the dispersion parameter
            a = disp_grid[t]
            logpostvec[t] = log_posterior(
                a,
                ycol,
                mu_hat_col,
                x,
                log_alpha_prior_mean[i],
                log_alpha_prior_sigmasq,
                usePrior,
                weights[:, i],
                useWeights,
                weightThreshold,
                useCR,
            )

        idxmax = np.argmax(logpostvec)
        a_hat = disp_grid[idxmax]
        disp_grid_fine = np.linspace(a_hat - delta, a_hat + delta, disp_grid_n)
        for t in range(disp_grid_n):
            a = disp_grid_fine[t]
            logpostvec[t] = log_posterior(
                a,
                ycol,
                mu_hat_col,
                x,
                log_alpha_prior_mean[i],
                log_alpha_prior_sigmasq,
                usePrior,
                weights[:, i],
                useWeights,
                weightThreshold,
                useCR,
            )

        idxmax = np.argmax(logpostvec)
        log_alpha[i] = disp_grid_fine[idxmax]

    return log_alpha


def fitDispGridWrapper(**kwargs):
    for k, v in kwargs.items():
        if np.any(np.isnan(v)):
            raise ValueError(f"argument {k} of fitDispGrid contains a NaN value")

    minLogAlpha = np.log(1e-8)
    maxLogAlpha = np.log(np.maximum(10, kwargs["y"].shape[0]))
    dispGrid = np.linspace(minLogAlpha, maxLogAlpha, 20)
    kwargs["mu_hat"] = kwargs["mu"]
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
    fit the negative binomial GLM
    note: the betas are on the natural log scale
    """
    y_m, y_n = y.shape
    x_p = x.shape[1]

    assert beta_mat.shape == (y_n, x_p)
    assert y.shape[0] == x.shape[0]
    assert nf.shape == y.shape
    assert lambda_.ndim == 1
    assert lambda_.shape[0] == x.shape[1], f"{lambda_.shape}, {x.shape}"

    beta_var_mat = np.zeros(beta_mat.shape)
    contrast_num = np.zeros(beta_mat.shape[0])
    contrast_denom = np.zeros(beta_mat.shape[0])
    hat_diagonals = np.zeros(y.shape)
    # bound the estimated count, as weights include 1/mu
    large = 30.0
    iter_ = np.zeros(y_n)
    deviance = np.zeros(y_n)
    ridge = np.diag(lambda_)
    for i in range(y_n):
        # if i % 100 == 0:
        #    checkUserInterrupt()
        nfcol = nf[:, i]
        ycol = y[:, i]
        beta_hat = beta_mat[i, :]
        mu_hat = nfcol * np.exp(x @ beta_hat)
        mu_hat = np.maximum(mu_hat, minmu)
        dev = 0.0
        dev_old = 0.0
        if useQR:
            # make an orthonormal design matrix including the ridge penalty
            for t in range(maxit):
                iter_[i] += 1
                if useWeights:
                    w_vec = weights[:, i] * mu_hat / (1.0 + alpha_hat[i] * mu_hat)
                    w_sqrt_vec = np.sqrt(w_vec)
                else:
                    w_vec = mu_hat / (1.0 + alpha_hat[i] * mu_hat)
                    w_sqrt_vec = np.sqrt(w_vec)
                # prepare matrices
                weighted_x_ridge = np.vstack([x * w_sqrt_vec[:, None], np.sqrt(ridge)])
                q, r = np.linalg.qr(weighted_x_ridge)
                big_w_diag = np.ones(y_m + x_p)
                big_w_diag[:y_m] = w_vec
                # big_w_sqrt = diagmat(sqrt(big_w_diag))
                z = np.log(mu_hat / nfcol) + (ycol - mu_hat) / mu_hat
                w_diag = w_vec.copy()
                z_sqrt_w = z * np.sqrt(w_diag)
                big_z_sqrt_w = np.zeros(y_m + x_p)
                big_z_sqrt_w[:y_m] = z_sqrt_w
                # IRLS with Q matrix for X
                gamma_hat = q.T @ big_z_sqrt_w
                beta_hat = np.linalg.solve(r, gamma_hat)
                if np.sum(np.abs(beta_hat) > large) > 0:
                    iter_[i] = maxit
                    break
                mu_hat = nfcol * np.exp(x @ beta_hat)
                mu_hat = np.maximum(mu_hat, minmu)
                dev = 0.0
                if useWeights:
                    dev -= 2.0 * np.sum(
                        weights[:, i]
                        * dnbinom_mu(ycol, 1.0 / alpha_hat[i], mu_hat, True)
                    )
                else:
                    dev -= 2.0 * np.sum(
                        dnbinom_mu(ycol, 1.0 / alpha_hat[i], mu_hat, True)
                    )

                conv_test = np.abs(dev - dev_old) / (np.abs(dev) + 0.1)
                if np.isnan(conv_test):
                    iter_[i] = maxit
                    break
                if t > 0 and conv_test < tol:
                    break
                dev_old = dev

        else:
            # use the standard design matrix x and matrix inversion
            for t in range(maxit):
                iter_[i] += 1
                if useWeights:
                    w_vec = weights[:, i] * mu_hat / (1.0 + alpha_hat[i] * mu_hat)
                    w_sqrt_vec = np.sqrt(w_vec)
                else:
                    w_vec = mu_hat / (1.0 + alpha_hat[i] * mu_hat)
                    w_sqrt_vec = np.sqrt(w_vec)

                z = np.log(mu_hat / nfcol) + (ycol - mu_hat) / mu_hat
                beta_hat = np.linalg.solve(
                    x.T @ (x.T * w_vec).T + ridge, x.T @ (z.T * w_vec).T
                )
                if np.sum(np.abs(beta_hat) > large) > 0:
                    iter_[i] = maxit
                    break
                mu_hat = nfcol * np.exp(x @ beta_hat)
                mu_hat = np.maximum(mu_hat, minmu)
                dev = 0.0
                if useWeights:
                    dev -= 2.0 * np.sum(
                        weights[:, i]
                        * dnbinom_mu(ycol, 1.0 / alpha_hat[i], mu_hat, True)
                    )
                else:
                    dev -= 2.0 * np.sum(
                        dnbinom_mu(ycol, 1.0 / alpha_hat[i], mu_hat, True)
                    )

                conv_test = np.abs(dev - dev_old) / (np.abs(dev) + 0.1)
                if np.isnan(conv_test):
                    iter_[i] = maxit
                    break
                if t > 0 and conv_test < tol:
                    break
                dev_old = dev

        deviance[i] = dev
        beta_mat[i, :] = beta_hat
        # recalculate w so that this is identical if we start with beta_hat
        if useWeights:
            w_vec = weights[:, i] * mu_hat / (1.0 + alpha_hat[i] * mu_hat)
            w_sqrt_vec = np.sqrt(w_vec)
        else:
            w_vec = mu_hat / (1.0 + alpha_hat[i] * mu_hat)
            w_sqrt_vec = np.sqrt(w_vec)

        hat_matrix_diag = np.zeros(x.shape[0])
        xw = x * w_sqrt_vec[:, None]
        xtwxr_inv = np.linalg.inv(x.T @ (x * w_vec[:, None]) + ridge)

        hat_matrix = xw @ xtwxr_inv @ xw.T
        hat_matrix_diag = np.diag(hat_matrix)

        hat_diagonals[:, i] = hat_matrix_diag
        # sigma is the covariance matrix for the betas
        sigma = xtwxr_inv @ x.T @ (x * w_vec[:, None]) @ xtwxr_inv
        contrast_num[i] = contrast.T @ beta_hat
        contrast_denom[i] = np.sqrt(contrast.T @ sigma @ contrast)
        beta_var_mat[i, :] = np.diag(sigma)

    return {
        "beta_mat": beta_mat,
        "beta_var_mat": beta_var_mat,
        "iter": iter_,
        "hat_diagonals": hat_diagonals,
        "contrast_num": contrast_num,
        "contrast_denom": contrast_denom,
        "deviance": deviance,
    }


def fitBetaWrapper(**kwargs):
    for k, v in kwargs.items():
        if np.any(np.isnan(v)):
            raise ValueError(f"argument {k} of fitBeta contains a NaN value")

    if "contrast" not in kwargs:
        kwargs["contrast"] = np.zeros(kwargs["x"].shape[1])
        kwargs["contrast"][0] = 1

    return fitBeta(**kwargs)
