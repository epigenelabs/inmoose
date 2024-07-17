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

# This file is based on the file 'R/fitFDist.R' of the Bioconductor limma package (version 3.55.1).

import numpy as np
from scipy.special import digamma, polygamma

from ..utils import LOGGER, lm_fit, ns


def fitFDist(x, df1, covariate=None):
    """
    Moment estimation of the parameters of a scaled F-distribution given one of the degrees of freedom.

    This function is called internally by :func:`eBayes` and :func:`squeezeVar`
    and is not usually called directly by a user.

    This function implements an algorithm proposed by [Smyth2004]_ and
    [Phipson2016]_. It estimates :code:`scale` and :code:`df2` under the
    assumption that :code:`x` is distributed as :code:`scale` times an
    F-distributed random variable on :code:`df1` and :code:`df2` degrees of
    freedom. The parameters are estimated using the method of moments,
    specifically from the mean and variance of the :code:`x` values on the
    log-scale.

    When :code:`covariate` is supplied, a spline curve trend will be estimated
    for the :code:`x` values and the estimation will be adjusted for this trend
    [Phipson2016]_.

    Arguments
    ---------
    x : array_like
        1-D array of positive values representing a sample from a scaled F-distribution
    df1 : array_like
        the first degrees of freedom of the F-distribution. Can be a single
        value or an array of the same length as :code:`x`.
    covariate :
        if not :code:`None`, the estimated scale value will depend on this
        numeric covariate.

    Returns
    -------
    dict
        a dictionnary with the following components:

        - :code:`"scale"`, a scale factor for F-distribution. An array if
          :code:`covariate` is not :code:`None`, a scalar otherwise.
        - :code:`"df2"`, the second degrees of freedom of the fitted F-distribution.
    """

    # Check x
    n = len(x)
    if n == 0:
        return {"scale": np.nan, "df2": np.nan}
    if n == 1:
        return {"scale": x, "df2": 0}

    # Check df1
    df1 = np.asarray(df1)
    ok = np.isfinite(df1) & (df1 > 1e-15)
    if df1.ndim == 0:
        if not ok:
            return {"scale": np.nan, "df2": np.nan}
        else:
            ok = np.full(n, True)
    else:
        if len(df1) != n:
            raise ValueError("x and df1 have different lengths")
    df1 = np.broadcast_to(df1, x.shape)

    # Check covariate
    if covariate is None:
        splinedf = 1
    else:
        covariate = np.asarray(covariate)
        if len(covariate) != n:
            raise ValueError("x and covariate must be of same length")
        if np.isnan(covariate).any():
            raise ValueError("NA covariate values not allowed")
        isfin = np.isfinite(covariate)
        if not isfin.all():
            if isfin.any():
                covariate[np.isneginf(covariate)] = np.min(covariate[isfin]) - 1
                covariate[np.isposinf(covariate)] = np.max(covariate[isfin]) + 1
            else:
                covariate = np.sign(covariate)

    # Remove missing or infinite or negative values and zero degrees of freedom
    ok = ok & np.isfinite(x) & (x > -1e-15)
    nok = ok.sum()
    if nok == 1:
        return {"scale": x[ok], "df2": 0}
    notallok = nok < n
    if notallok:
        x = x[ok]
        df1 = df1[ok]
        if covariate is not None:
            covariate_notok = covariate[~ok]
            covariate = covariate[ok]

    # Set df for spline trend
    if covariate is not None:
        splinedf = 1 + (nok >= 3) + (nok >= 6) + (nok >= 30)
        splinedf = min(splinedf, len(np.unique(covariate)))
        # If covariate takes only one unique value or insufficient observations, recall
        # with None covariate
        if splinedf < 2:
            out = fitFDist(x=x, df1=df1)
            out["scale"] = np.full(n, out["scale"])
            return out

    # Avoid exactly zero values
    x = np.maximum(x, 0)
    m = np.median(x)
    if m == 0:
        LOGGER.warning(
            "More than half of residual variances are exactly zero: eBayes unreliable"
        )
        m = 1
    else:
        if (x == 0).any():
            LOGGER.warning(
                "Zero sample variances detected, have been offset away from zero"
            )
    x = np.maximum(x, 1e-5 * m)

    # Better to work on with log(F)
    z = np.log(x)
    e = z - digamma(df1 / 2) + np.log(df1 / 2)

    if covariate is None:
        emean = e.mean()
        evar = np.sum((e - emean) ** 2 / (nok - 1))
    else:
        try:
            design = ns(covariate, df=splinedf, include_intercept=True)
        except:  # noqa: E722
            raise RuntimeError("Problem with covariate")
        fit = lm_fit(design.basis, e)
        if notallok:
            design2 = design.predict(newx=covariate_notok)
            emean = np.zeros(n)
            emean[ok] = fit.fitted_values
            emean[~ok] = design2.basis @ fit.coefficients
        else:
            emean = fit.fitted_values
        evar = (fit.effects[fit.rank :] ** 2).mean()

    # Estimate scale and df2
    evar = evar - polygamma(1, df1 / 2).mean()
    if evar > 0:
        df2 = 2 * trigammaInverse(evar)
        s20 = np.exp(emean + digamma(df2 / 2) - np.log(df2 / 2))
    else:
        df2 = np.inf
        if covariate is None:
            # Use simple pooled variance, which is MLE of the scale in this case.
            # Versions of limma before Jan 2017 returned the limiting value of the
            # evar>0 estimate, which is larger.
            s20 = x.mean()
        else:
            s20 = np.exp(emean)

    return {"scale": s20, "df2": df2}


def trigammaInverse(x):
    """
    Solve trigamma(y) = x for y

    Arguments
    ---------
    x : array_like

    Returns
    -------
    ndarray
        trigamma inverses for each value of :code:`x`
    """
    x = np.asarray(x)

    # Treat out-of-range values as special cases
    omit = np.isnan(x)
    if omit.any():
        y = x
        if (~omit).any():
            y[~omit] = trigammaInverse(x[~omit])
        return y

    omit = x < 0
    if omit.any():
        y = x
        y[omit] = np.nan
        LOGGER.warning("NaNs produced")
        if (~omit).any():
            y[~omit] = trigammaInverse(x[~omit])
        return y

    omit = x > 1e7
    if omit.any():
        y = x
        y[omit] = 1 / np.sqrt(x[omit])
        if (~omit).any():
            y[~omit] = trigammaInverse(x[~omit])
        return y

    omit = x < 1e-6
    if omit.any():
        y = x
        with np.errstate(divide="ignore"):
            y[omit] = 1 / x[omit]
        if (~omit).any():
            y[~omit] = trigammaInverse(x[~omit])
        return y

    # Newton's method
    # 1/trigamma(y) is convex, nearly linear and strictly > y-0.5,
    # so iteration to solve 1/x = 1/trigamma is monotonically convergent
    y = 0.5 + 1 / x
    it = 0
    while True:
        it += 1
        tri = polygamma(1, y)
        dif = tri * (1 - tri / x) / polygamma(2, y)
        y += dif
        if np.max(-dif / y) < 1e-8:
            break
        if it > 50:
            LOGGER.warning("Iteration limit exceeded")
            break
    return y
