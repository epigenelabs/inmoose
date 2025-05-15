# -----------------------------------------------------------------------------
# Copyright (C) 2022-2023 M. Colange

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

import numpy as np
from scipy.stats import nbinom, norm, t

from .stats_cpp import nbinom_logpmf


def rnbinom(n, size, mu, seed=None):
    r"""mimic R rnbinom function, to draw samples from a Negative Binomial distribution.

    The (:math:`size`, :math:`p`) parameterization used in R is the same as in scipy.stats:
    :math:`p = 1 / (1 + \mu/size) = size / (size + \mu)`.

    Arguments
    ---------
    n : int or tuple of ints
        shape of the output. If n = (n1, n2, ..., np) then n1*n2*...*np random samples are drawn.
    size : float or array-like
        size parameter of the Negative Binomial distribution.
        all values must be positive
    mu : float or array-like
        mean parameter of the Negative Binomial distribution
        all values must be positive
    seed : int, optional
        pass a seed to the underlying RNG. If `None`, then the RNG is seeded using unpredictable entropy from the system.
        See the documentation of scipy.stats about RNG seeding for more details.
    """
    p = size / (size + mu)
    return nbinom(size, p).rvs(n, random_state=seed)


@np.errstate(divide="ignore")
def dnbinom_mu(x, size, mu, log=False):
    """mimic R dnbinom_mu function, to compute the density function of a Negative Binomial distribution.

    The (size, prob) parameterization used in R dnbinom is the same as in scipy.stats:
        mu = size * (1 - prob) / prob = size * (1/prob - 1)
    so
        mu / size + 1 = 1/prob
    hence
        prob = 1 / (1 + mu/size) = size / (size + mu)

    Arguments
    ---------
    x : array-like
        points where the density function is evaluated
    size : float or array-like
        size parameter of the Negative Binomial distribution.
        all values must be positive
    mu : float or array-like
        mean parameter of the Negative Binomial distribution
        all values must be positive
    log: bool, optional
        switch to output the density in normal (log=False) or log (log=True) scale
    """
    p = size / (size + mu)
    if log:
        return nbinom_logpmf(x, size, p)
    else:
        return nbinom.pmf(x, n=size, p=p)


def dnorm(x, mean, sd, log=False):
    """mimic R dnorm function, to compute the density function of a normal distribution.

    Arguments
    ---------
    x : array-like
        points where the density function is evaluated
    mean : float or array-like
        mean of the normal distribution
    sd : float or array-like
        standard deviation of the normal distribution
    log : bool, optional
        switch to output the density in normal (log=False) or log (log=True) scale
    """
    if log:
        return norm.logpdf(x, loc=mean, scale=sd)
    else:
        return norm.pdf(x, loc=mean, scale=sd)


def pnorm(q, mean=0, sd=1, lower_tail=True, log_p=False):
    """mimic R pnorm function, to compute the distribution function of a normal distribution.

    Arguments
    ---------
    q : array-like
        points where the distribution function is evaluated
    mean : float or array-like
        mean of the normal distribution
    sd : float or array-like
        standard deviation of the normal distribution
    log: bool, optional
        switch to output the distribution in normal (log=False) or log (log=True) scale
    lower_tail : bool, optional
        switch to output P[X <= q] (lower_tail=True) or P[X > q] (lower_tail=False)
    """
    if lower_tail and log_p:
        f = norm.logcdf
    elif lower_tail and not log_p:
        f = norm.cdf
    elif not lower_tail and log_p:
        f = norm.logsf
    elif not lower_tail and not log_p:
        f = norm.sf

    return f(q, loc=mean, scale=sd)


def pt(q, df, lower_tail=True, log_p=False):
    """mimic R pt function, to compute the distribution function of a Student t distribution.

    Arguments
    ---------
    q : array-like
        points where the distribution function is evaluated
    df : int or array-like
        number of degrees of freedom of the Student t distribution
    log: bool, optional
        switch to output the distribution in normal (log=False) or log (log=True) scale
    lower_tail : bool, optional
        switch to output P[X <= q] (lower_tail=True) or P[X > q] (lower_tail=False)
    """
    if lower_tail and log_p:
        f = t.logcdf
    elif lower_tail and not log_p:
        f = t.cdf
    elif not lower_tail and log_p:
        f = t.logsf
    elif not lower_tail and not log_p:
        f = t.sf

    return f(q, df=df)
