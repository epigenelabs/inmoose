# -----------------------------------------------------------------------------
# Copyright (C) 2024-2025 Gordon Smyth, Lizhong Chen, Aaron Lun
# Copyright (C) 1996-2025 Robert Gentleman, Ross Ihaka, R Core Team
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed without any warranty. See
# <https://www.gnu.org/licenses/> for the GNU General Public License.
# -----------------------------------------------------------------------------

# Based on limma's R/fitFDistUnequalDF1.R (12 July 2025), R/loessFit.R,
# R/weightedLowess.R and src/weighted_lowess.c. Only non-robust estimation
# (one LOWESS iteration) is implemented.

import warnings

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln, polygamma


def fitFDistUnequalDF1(
    x, df1, covariate=None, span=None, robust=False, prior_weights=None
):
    """Estimate a scaled F-distribution allowing unequal numerator df.

    The log-scale mean is weighted by inverse trigamma precision. The
    denominator df is estimated by profile likelihood, as in current R limma.
    With a covariate, the mean is fitted by weighted LOWESS.

    Parameters
    ----------
    x : array_like
        One-dimensional sample variances. Missing values receive zero weight;
        nonpositive values are offset to 1e-12 times the positive median.
    df1 : float or array_like
        Residual degrees of freedom, scalar or the same length as ``x``.
        Values below 0.01 receive zero weight. NaNs are not allowed.
    covariate : array_like, optional
        Numeric trend covariate of the same length as ``x``; no NaNs allowed.
    span : float, optional
        LOWESS span. By default, limma's sample-size-dependent span is used.
    robust : bool
        Robust estimation is not implemented; True raises NotImplementedError.
    prior_weights : array_like, optional
        Nonnegative observation weights of the same length as ``x``.

    Returns
    -------
    dict
        ``scale`` (scalar, or an array for a trend) and ``df2`` (scalar).
        Fewer than two informative positive variances give NaN estimates.
    """
    if robust:
        raise NotImplementedError(
            "Robust estimation in fitFDistUnequalDF1 is not implemented"
        )
    x = np.array(x, dtype=float, copy=True)
    n = len(x)
    df1 = np.atleast_1d(np.array(df1, dtype=float, copy=True))
    if df1.size not in (1, n):
        raise ValueError("x and df1 are different lengths")
    if np.isnan(df1).any():
        raise ValueError("NA df1 values")
    df1 = np.broadcast_to(df1, x.shape).copy()
    if covariate is not None:
        covariate = np.asarray(covariate, dtype=float)
        if covariate.shape != x.shape:
            raise ValueError("x and covariate are different lengths")
        if np.isnan(covariate).any():
            raise ValueError("NA covariate values")
    if prior_weights is not None:
        prior_weights = np.array(prior_weights, dtype=float, copy=True)
        if prior_weights.shape != x.shape:
            raise ValueError("x and prior_weights are different lengths")
        if np.isnan(prior_weights).any():
            raise ValueError("prior_weights contain NA values")
        if (prior_weights < 0).any():
            raise ValueError("prior_weights are negative")

    missing = np.isnan(x)
    small_df = df1 < 0.01
    if missing.any() or small_df.any():
        if prior_weights is None:
            prior_weights = np.ones(n)
        prior_weights[missing | small_df] = 0
        x[missing] = 0
        df1[small_df] = 1
    weighted = prior_weights is not None
    informative = x > 0
    if weighted:
        informative &= prior_weights != 0
    n_informative = informative.sum()
    if n_informative < 2:
        return {"scale": np.nan, "df2": np.nan}
    if n_informative == 2:
        covariate = None
    # R clears prior.weights for two informative values without clearing its
    # weighted flag: the mean is then NaN and the likelihood is identically 0.
    empty_weights = weighted and n_informative == 2

    xpos = np.maximum(x, 1e-12 * np.median(x[informative]))
    d1 = df1 / 2
    e = np.log(xpos) + _logmdigamma(d1)
    w = 1 / polygamma(1, d1)
    if weighted:
        w *= prior_weights
    if empty_weights:
        emean = np.nan
    elif covariate is None:
        emean = np.sum(w * e) / np.sum(w)
    else:
        if span is None:
            span = min(0.3 + 0.7 * (500 / n) ** (1 / 3), 1)
        emean = _loess_fit(e, covariate, w / np.quantile(w, 0.75), span)

    d1x = d1 * xpos

    def minus_twice_loglik(par):
        if empty_weights:
            return 0.0
        d2 = par / (1 - par)
        d2s20 = d2 * np.exp(emean - _logmdigamma(d2))
        terms = (
            -(d1 + d2) * np.log1p(d1x / d2s20)
            - d1 * np.log(d2s20)
            + gammaln(d1 + d2)
            - gammaln(d2)
        )
        if weighted:
            terms *= prior_weights
        value = -2 * np.sum(terms)
        # R's optimize replaces nonfinite function values with DBL_MAX.
        return value if np.isfinite(value) else np.finfo(float).max

    # R optimize uses Brent's bounded algorithm and eps**0.25 tolerance.
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        out = minimize_scalar(
            minus_twice_loglik,
            bounds=(0.5, 0.9998),
            method="bounded",
            options={"xatol": np.finfo(float).eps ** 0.25},
        )
        d2 = out.x / (1 - out.x)
        scale = np.exp(emean - _logmdigamma(d2))
    if out.fun == np.finfo(float).max:
        warnings.warn("Nonfinite likelihood in fitFDistUnequalDF1", RuntimeWarning)
    return {"scale": scale, "df2": 2 * d2}


def _logmdigamma(x):
    """Stable log(x) - digamma(x), using statmod's recurrence and expansion."""
    x = np.asarray(x, dtype=float)
    z = np.where(x < 5, x + 5, x)
    r = 1 / z**2
    coefficients = [
        3617 / 8160, -1 / 12, 691 / 32760, -1 / 132,
        1 / 240, -1 / 252, 1 / 120, -1 / 12,
    ]
    result = 1 / (2 * z) - r * np.polyval(coefficients, r)
    small = x < 5
    if np.any(small):
        correction = np.log(x[small] / (x[small] + 5))
        for i in range(5):
            correction += 1 / (x[small] + i)
        result = np.array(result, copy=True)
        result[small] += correction
    return result


def _loess_fit(y, x, weights, span):
    """The single-iteration loessFit path used by fitFDistUnequalDF1."""
    fitted = np.full(len(y), np.nan)
    obs = np.isfinite(x) & np.isfinite(y)
    x, y, weights = x[obs], y[obs], weights[obs]
    n = len(y)
    if not n:
        return fitted
    if span < 1 / n:
        fitted[obs] = y
        return fitted
    weights = np.clip(np.nan_to_num(weights, nan=0), 1e-8, 1e2)
    if np.ptp(weights) < 1e-15:
        fitted[obs] = _lowess(x, y, span)
    elif n < 4 + 1 / span:
        design = np.column_stack((np.ones(n), x))
        rootw = np.sqrt(weights)
        coef = np.linalg.lstsq(design * rootw[:, None], y * rootw, rcond=None)[0]
        fitted[obs] = design @ coef
    else:
        fitted[obs] = _weighted_lowess(x, y, weights, span)
    return fitted


def _lowess(x, y, span):
    """R stats::lowess, iter=0 and default delta, including tied covariates.

    Based on R's src/library/stats/src/lowess.c (Cleveland's algorithm).
    """
    if len(x) < 2:
        return y.copy()
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    n = len(x)
    width = max(2, min(n, int(span * n + 1e-7)))
    left, right = 0, width - 1
    last, i = -1, 0
    fitted = np.empty(n)
    spread = np.ptp(x)
    delta = 0.01 * spread
    while True:
        while right < n - 1 and x[i] - x[left] > x[right + 1] - x[i]:
            left += 1
            right += 1
        distance = max(x[i] - x[left], x[right] - x[i])
        xx = x[left:]
        r = np.abs(xx - x[i])
        w = np.zeros(len(xx))
        w[r <= 0.001 * distance] = 1
        inside = (r > 0.001 * distance) & (r <= 0.999 * distance)
        if inside.any():
            w[inside] = (1 - (r[inside] / distance) ** 3) ** 3
        w /= w.sum()
        if distance > 0:
            center = np.sum(w * xx)
            variance = np.sum(w * (xx - center) ** 2)
            if np.sqrt(variance) > 0.001 * spread:
                w *= 1 + (x[i] - center) / variance * (xx - center)
        fitted[i] = np.sum(w * y[left:])
        if last < i - 1:
            alpha = (x[last + 1 : i] - x[last]) / (x[i] - x[last])
            fitted[last + 1 : i] = alpha * fitted[i] + (1 - alpha) * fitted[last]
        last = i
        cut = x[last] + delta
        j = last + 1
        while j < n and x[j] <= cut:
            if x[j] == x[last]:
                fitted[j] = fitted[last]
                last = j
            j += 1
        if last == n - 1:
            break
        i = max(last + 1, j - 1)
    result = np.empty(n)
    result[order] = fitted
    return result


def _weighted_lowess(x, y, weights, span):
    """limma weightedLowess with npts=200 and no robustness iterations.

    Prior weights determine both the neighbourhood width and local regression.
    Fit at seed points and interpolate between them, as in limma's C kernel.
    """
    order = np.argsort(x, kind="stable")
    x, y, weights = x[order], y[order], weights[order]
    n = len(x)
    delta = 0
    if n > 200:
        cumrange = np.cumsum(np.sort(np.diff(x)))
        clusters = np.arange(200)
        delta = np.min(cumrange[n - 2 - clusters] / (200 - clusters))
    seeds = [0]
    for j in range(1, n - 1):
        if x[j] - x[seeds[-1]] > delta:
            seeds.append(j)
    seeds.append(n - 1)
    # Sequential summation matches the C span-boundary decisions.
    span_weight = sum(weights) * span
    fitted = np.empty(n)
    for j in seeds:
        left = right = j
        current_weight = weights[j]
        while current_weight < span_weight and (left > 0 or right < n - 1):
            if right == n - 1 or (
                left > 0 and x[j] - x[left - 1] < x[right + 1] - x[j]
            ):
                left -= 1
                current_weight += weights[left]
            else:
                right += 1
                current_weight += weights[right]
        distance = max(x[j] - x[left], x[right] - x[j])
        while left > 0 and x[left] == x[left - 1]:
            left -= 1
        while right < n - 1 and x[right] == x[right + 1]:
            right += 1
        xx, yy = x[left : right + 1], y[left : right + 1]
        w = weights[left : right + 1].copy()
        if distance >= 1e-7:
            w *= (1 - (np.abs(x[j] - xx) / distance) ** 3) ** 3
        ymean = np.sum(w * yy) / np.sum(w)
        fitted[j] = ymean
        if distance >= 1e-7:
            xmean = np.sum(w * xx) / np.sum(w)
            variance = np.sum(w * (xx - xmean) ** 2)
            if variance >= 1e-7:
                slope = np.sum(w * (xx - xmean) * (yy - ymean)) / variance
                fitted[j] = slope * x[j] + (ymean - slope * xmean)
    subrange = np.ptp(x) / n
    for left, right in zip(seeds[:-1], seeds[1:]):
        if right - left > 1:
            distance = x[right] - x[left]
            if distance > 1e-7 * subrange:
                slope = (fitted[right] - fitted[left]) / distance
                intercept = fitted[right] - slope * x[right]
                fitted[left + 1 : right] = slope * x[left + 1 : right] + intercept
            else:
                fitted[left + 1 : right] = (fitted[left] + fitted[right]) / 2
    result = np.empty(n)
    result[order] = fitted
    return result
