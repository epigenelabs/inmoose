# -----------------------------------------------------------------------------
# Copyright (C) 2008-2022 Yunshun Chen, Aaron TL Lun, Davis J McCarthy, Matthew E Ritchie, Belinda Phipson, Yifang Hu, Xiaobei Zhou, Mark D Robinson, Gordon K Smyth
# Copyright (C) 2022-2023 Maximilien Colange

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

# This file is a Python port of the original C++ code from the files
# 'src/R_fit_levenberg.cpp' and 'src/glm_levenberg.cpp' of the Bioconductor
# edgeR package (version 3.38.4).

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve

from .nbdev import nb_deviance

low_value = 1e-10
supremely_low_value = 1e-13
ridiculously_low_value = 1e-100


def fit_levenberg(y, offset, disp, weights, design, beta, tol, maxit):
    """
    Arguments
    ---------
    y : array_like
        matrix of counts
    offset : array_like
        offsets, same shape as :code:`y`
    disp : array_like
        dispersions, same shape as :code:`y`
    weights : array_like
        weights, same shape as :code:`y`
    design : array_like
        design matrix, as many rows as columns in :code:`y`
    beta : array_like
        initial values of the beta, as many rows as in :code:`y`, as many
        columns as in :code:`design`
    tol : float
        tolerance for convergence
    maxit : int
        maximal number of iterations

    Returns
    -------
    ndarray
        fitted beta, same shape as :code:`design`
    ndarray
        fitted mu, same shape as :code:`y`
    ndarray
        genewise deviance
    ndarray
        genewise number of iterations
    ndarray
        whether convergence was reached, genewise
    """

    assert offset.shape == y.shape
    assert disp.shape == y.shape
    assert weights.shape == y.shape
    assert design.shape[0] == y.shape[1]
    assert beta.shape == (y.shape[0], design.shape[1])

    out_beta = beta.copy()
    mu = np.zeros(y.shape)
    dev = np.zeros(y.shape[0])
    iter_ = np.zeros(y.shape[0])
    conv = np.repeat(False, y.shape[0])

    # We compute mu based on beta. Returning if there are no coeffs
    mu = np.exp(out_beta @ design.T + offset)
    dev = nb_deviance(y, mu, weights, disp)

    # for all-zero libraries, there is really no point continuing
    ymax = np.max(y, axis=1)
    out_beta[ymax < low_value, :] = np.nan
    mu[ymax < low_value, :] = 0
    conv[ymax < low_value] = False
    dev[ymax < low_value] = 0
    iter_[ymax < low_value] = 0

    remaining = ~(ymax < low_value)

    # iterating using reweighted least squares
    for i in range(maxit):
        if not remaining.any():
            break

        iter_[remaining] += 1

        # Here we set up the matrix XtWX i.e. the Fisher information matrix.
        # X is the design matrix and W is a diagonal matrix with the working
        # weights for each observation (i.e. library). The working weights are
        # part of the first derivative of the log-likelihood for a given
        # coefficient, multiplied by any user-specified weights. When multiplied
        # by two covariates in the design matrix, you get the Fisher information
        # (i.e. variance of the log-likelihood) for that pair. This takes the
        # role of the second derivative of the log-likelihood. The working
        # weights are formed by taking the reciprocal of the product of the
        # variance (in terms of the mean) and the square of the derivative of
        # the link function.

        # We also set up the actual derivative of the log likelihoods in 'dl'.
        # This is done by multiplying each covariate by the difference between
        # the mu and observation and dividing by the variance and derivative of
        # the link function. This is then summed across all observations for
        # each coefficient. The aim is to solve (XtWX)(dbeta)=dl for 'dbeta'.
        # As XtWX is the second derivative, and dl is the first, you can see
        # that we are effectively performing a multivariate Newton-Raphson
        # procedure with 'dbeta' as the step.

        denom = 1 + mu * disp
        deriv = (y - mu) / denom * weights

        working_weights = mu / denom * weights
        xtwx = (design.T * np.expand_dims(working_weights, axis=-2)) @ design

        dl = deriv @ design
        assert dl.shape == (y.shape[0], design.shape[1])
        dbeta = np.zeros(dl.shape)
        maxinfo = np.max(np.diagonal(xtwx, axis1=-1, axis2=-2), axis=1)
        if i == 0:
            lambda_ = np.maximum(maxinfo * 1e-6, supremely_low_value)

        # Levenberg/Marquardt damping reduces step size until the deviance
        # increases or no step can be found that increases the deviance. In
        # short, increases in the deviance are enforced to avoid problems with
        # convergence.
        lev = np.repeat(0, y.shape[0])
        low_dev = np.repeat(False, y.shape[0])

        lev_rem = remaining.copy()
        while lev_rem.any():
            lev[lev_rem] += 1

            for tag in range(y.shape[0]):
                if not lev_rem[tag]:
                    continue

                while True:
                    # We add lambda_ to the diagonal. This reduces the step size
                    # as the second derivative is increased.
                    xtwx_copy = xtwx[tag] + lambda_[tag] * np.eye(xtwx.shape[1])

                    # Cholesky decomposition, and then use the decomposition to
                    # solve for dbeta in (XtWX)dbeta = dl.
                    try:
                        xtwx_cho = cho_factor(xtwx_copy)
                        break
                    except LinAlgError:
                        # If it fails, it MUST mean that the matrix is singular
                        # due to numerical imprecision as all the diagonal
                        # entries of the XtWX matrix must be positive. This
                        # occurs because of fitted values being exactly zero;
                        # thus, the coefficients attempt to converge to
                        # negative infinity. This generally forces the step
                        # size to be larger (i.e. lambda_ lower) in order to
                        # get to infinity faster (which is impossible). Low
                        # lambda_ leads to numerical instability and effective
                        # singularity. To solve this, we actually increase
                        # lambda_; this avoids code breakage to give the other
                        # coefficients a chance to converge. Failure of
                        # convergence for the zero fitted values is not a
                        # problem as the change in deviance from small ->
                        # smaller coefficients is not that great when the true
                        # value is negative infinity.
                        lambda_[tag] *= 10
                        if lambda_[tag] <= 0:
                            # just to make sure it actually increases
                            lambda_[tag] = ridiculously_low_value

                dbeta[tag, :] = cho_solve(xtwx_cho, dl[tag, :])

            # Updating beta and the means. 'dbeta' stores 'Y' from the
            # solution of (X*VX)Y=dl, corresponding to a NR step.
            beta_new = np.zeros(beta.shape)
            mu_new = np.zeros(mu.shape)
            beta_new[lev_rem, :] = out_beta[lev_rem, :] + dbeta[lev_rem, :]
            mu_new[lev_rem, :] = np.exp(
                beta_new[lev_rem, :] @ design.T + offset[lev_rem, :]
            )

            # Checking if the deviance has decreased or if it is too small to
            # care about. Either case is good and means that we will be using
            # the updated fitted values and coefficients. Otherwise, if we have
            # to repeat the inner loop, then we want to do so from the original
            # values (as we will be scaling lambda up so we want to retake the
            # step from where we were before). This is why we do not modify the
            # values in-place until we are sure we want to take the step.
            dev_new = nb_deviance(y, mu_new, weights, disp)
            with np.errstate(invalid="ignore"):
                low_dev[(dev_new / ymax) < supremely_low_value] = True
            low_tags = lev_rem & ((dev_new <= dev) | low_dev)
            out_beta[low_tags, :] = beta_new[low_tags, :]
            mu[low_tags, :] = mu_new[low_tags, :]
            dev[low_tags] = dev_new[low_tags]
            lev_rem &= ~low_tags

            # Increasing lambda_, to increase damping. Again, we have to
            # make sure it is not zero.
            lambda_[lev_rem] *= 2
            lambda_[lev_rem & (lambda_ <= 0)] = ridiculously_low_value

            # Excessive damping; steps get so small that it is pointless to
            # continue.
            with np.errstate(divide="ignore"):
                conv[lev_rem & ((lambda_ / maxinfo) > (1 / supremely_low_value))] = True
            lev_rem &= ~conv

        # Terminating if we failed, if divergence from the exact solution is
        # acceptably low (cross-product of dbeta with the log-likelihood
        # derivative) or if the actual deviance of the fit is acceptably low.
        remaining &= ~conv
        remaining &= ~low_dev
        # NB: np.diag(A @ B.T) is equivalent to np.sum(A * B, axis=1)
        divergence = np.sum(dl * dbeta, axis=1)
        remaining &= ~(divergence < tol)

        # If we quit the inner Levenberg loop immediately and survived all the
        # break conditions above, that means that deviance is decreasing
        # substantially. Thus, we need larger steps to get there faster. To do
        # so, we decrease the damping factor. Note that this only applies if we
        # did not decrease the damping factor in the inner Levenberg loop, as
        # that would indicate that we need to slow down.
        lambda_[remaining & (lev == 1)] /= 10

    return (out_beta, mu, dev, iter_, conv)


# for reference, but it is likely dead code
def autofill(design, beta, offset):
    return np.exp(design @ beta + offset)
