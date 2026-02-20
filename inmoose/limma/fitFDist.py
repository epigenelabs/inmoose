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

import math
import warnings

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import stats
from scipy.optimize import brentq
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
        a dictionary with the following components:

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


def _trimmed_mean(values: np.ndarray, trim: float) -> float:
    if trim <= 0:
        return float(np.mean(values))
    n = len(values)
    if n == 0:
        return float("nan")
    k = int(math.floor(trim * n))
    if k == 0:
        return float(np.mean(values))
    if 2 * k >= n:
        return float(np.mean(values))
    vals = np.sort(values)
    return float(np.mean(vals[k : n - k]))


def _loess_fit(y: np.ndarray, x: np.ndarray, span: float = 0.4) -> dict[str, np.ndarray]:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return {"fitted": np.asarray([]), "residuals": np.asarray([])}
    k = max(2, int(math.ceil(span * n)))
    k = min(k, n)
    fitted = np.empty(n, dtype=float)
    for i in range(n):
        dist = np.abs(x - x[i])
        idx = np.argpartition(dist, k - 1)[:k]
        dmax = dist[idx].max()
        if dmax <= 0:
            fitted[i] = y[i]
            continue
        w = (1 - (dist[idx] / dmax) ** 3) ** 3
        x_centered = x[idx] - x[i]
        X = np.vstack([np.ones_like(x_centered), x_centered]).T
        XT_W = X.T * w
        try:
            beta = np.linalg.pinv(XT_W @ X) @ (XT_W @ y[idx])
            fitted[i] = beta[0]
        except np.linalg.LinAlgError:
            fitted[i] = np.average(y[idx], weights=w)
    return {"fitted": fitted, "residuals": y - fitted}


def _gauss_quad_uniform(n: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = leggauss(n)
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    return nodes, weights


def _f_ppf(p: np.ndarray, df1: float, df2: float) -> np.ndarray:
    if np.isinf(df2):
        return stats.chi2.ppf(p, df1) / df1
    return stats.f.ppf(p, df1, df2)


def _f_isf(p: np.ndarray, df1: float, df2: float) -> np.ndarray:
    if np.isinf(df2):
        return stats.chi2.isf(p, df1) / df1
    return stats.f.isf(p, df1, df2)


def _f_logcdf(x: np.ndarray, df1: np.ndarray, df2: float) -> np.ndarray:
    if np.isinf(df2):
        return stats.chi2.logcdf(np.asarray(x) * df1, df1)
    return stats.f.logcdf(x, df1, df2)


def _f_logsf(x: np.ndarray, df1: np.ndarray, df2: float) -> np.ndarray:
    if np.isinf(df2):
        return stats.chi2.logsf(np.asarray(x) * df1, df1)
    return stats.f.logsf(x, df1, df2)


def _f_pdf(x: np.ndarray, df1: float, df2: float) -> np.ndarray:
    if np.isinf(df2):
        return stats.chi2.pdf(df1 * x, df1) * df1
    return stats.f.pdf(x, df1, df2)


def fitFDistRobustly(
    x,
    df1,
    covariate=None,
    winsor_tail_p=(0.05, 0.1),
    trace: bool = False,
):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return {"scale": np.nan, "df2": np.nan, "df2.shrunk": np.nan}
    if n == 2:
        return fitFDist(x=x, df1=df1, covariate=covariate)

    df1 = np.asarray(df1, dtype=float)
    if df1.size not in (1, n):
        raise ValueError("x and df1 are different lengths")
    if covariate is not None:
        covariate = np.asarray(covariate, dtype=float)
        if len(covariate) != n:
            raise ValueError("x and covariate are different lengths")
        if not np.all(np.isfinite(covariate)):
            raise ValueError("covariate contains NA or infinite values")

    ok = ~np.isnan(x) & np.isfinite(df1) & (df1 > 1e-6)
    if not np.all(ok):
        df2_shrunk = np.array(x, copy=True)
        x_ok = x[ok]
        df1_ok = df1 if df1.size == 1 else df1[ok]
        cov_ok = covariate[ok] if covariate is not None else None
        cov_bad = covariate[~ok] if covariate is not None else None
        fit = fitFDistRobustly(
            x=x_ok,
            df1=df1_ok,
            covariate=cov_ok,
            winsor_tail_p=winsor_tail_p,
            trace=trace,
        )
        df2_shrunk[ok] = fit["df2.shrunk"]
        df2_shrunk[~ok] = fit["df2"]
        if covariate is None:
            scale = fit["scale"]
        else:
            scale = np.array(x, copy=True)
            scale[ok] = fit["scale"]
            order = np.argsort(cov_ok)
            cov_sorted = cov_ok[order]
            log_scale = np.log(np.asarray(fit["scale"])[order])
            interp = np.interp(
                cov_bad,
                cov_sorted,
                log_scale,
                left=log_scale[0],
                right=log_scale[-1],
            )
            scale[~ok] = np.exp(interp)
        return {"scale": scale, "df2": fit["df2"], "df2.shrunk": df2_shrunk}

    m = np.median(x)
    if m <= 0:
        raise ValueError("Variances are mostly <= 0")
    i_small = x < m * 1e-12
    if np.any(i_small):
        nzero = int(np.sum(i_small))
        if nzero == 1:
            warnings.warn(
                "One very small variance detected, has been offset away from zero",
                stacklevel=2,
            )
        else:
            warnings.warn(
                f"{nzero} very small variances detected, have been offset away from zero",
                stacklevel=2,
            )
        x[i_small] = m * 1e-12

    NonRobust = fitFDist(x=x, df1=df1, covariate=covariate)

    winsor_tail_p = np.resize(np.asarray(winsor_tail_p, dtype=float), 2)
    prob = winsor_tail_p.copy()
    prob[1] = 1 - winsor_tail_p[1]
    if np.all(winsor_tail_p < 1 / n):
        NonRobust["df2.shrunk"] = np.resize(NonRobust["df2"], n)
        return NonRobust

    if df1.size > 1:
        df1max = np.max(df1)
        idx = df1 < (df1max - 1e-14)
        if np.any(idx):
            s = NonRobust["scale"] if covariate is None else np.asarray(NonRobust["scale"])[idx]
            f = x[idx] / s
            df2 = NonRobust["df2"]
            pupper = _f_logsf(f, df1[idx], df2)
            plower = _f_logcdf(f, df1[idx], df2)
            up = pupper < plower
            if np.any(up):
                f[up] = _f_isf(np.exp(pupper[up]), df1max, df2)
            if np.any(~up):
                f[~up] = _f_ppf(np.exp(plower[~up]), df1max, df2)
            x[idx] = f * s
            df1 = df1max
        else:
            df1 = float(df1[0])
    elif df1.size == 1:
        df1 = float(df1[0])

    z = np.log(x)
    if covariate is None:
        ztrend = _trimmed_mean(z, winsor_tail_p[1])
        zresid = z - ztrend
    else:
        lo = _loess_fit(z, covariate, span=0.4)
        ztrend = lo["fitted"]
        zresid = lo["residuals"]

    zrq = np.quantile(zresid, prob)
    zwins = np.clip(zresid, zrq[0], zrq[1])
    zwmean = float(np.mean(zwins))
    zwvar = float(np.var(zwins, ddof=1))
    if trace:
        print("Variance of Winsorized Fisher-z", zwvar)

    quad_nodes, quad_weights = _gauss_quad_uniform(128)

    def winsorized_moments(df1_val: float, df2_val: float) -> dict[str, float]:
        fq = _f_ppf(np.asarray([winsor_tail_p[0], 1 - winsor_tail_p[1]]), df1_val, df2_val)
        zq = np.log(fq)
        q = fq / (1 + fq)
        nodes = q[0] + (q[1] - q[0]) * quad_nodes
        fnodes = nodes / (1 - nodes)
        znodes = np.log(fnodes)
        f_pdf = _f_pdf(fnodes, df1_val, df2_val) / (1 - nodes) ** 2
        q21 = q[1] - q[0]
        m_val = q21 * np.sum(quad_weights * f_pdf * znodes) + np.sum(zq * winsor_tail_p)
        v_val = q21 * np.sum(quad_weights * f_pdf * (znodes - m_val) ** 2) + np.sum(
            (zq - m_val) ** 2 * winsor_tail_p
        )
        return {"mean": float(m_val), "var": float(v_val)}

    mom_inf = winsorized_moments(df1, math.inf)
    funvalInf = math.log(zwvar / mom_inf["var"])
    if funvalInf <= 0:
        df2 = math.inf
        ztrendcorrected = ztrend + zwmean - mom_inf["mean"]
        s20 = np.exp(ztrendcorrected)
        Fstat = np.exp(z - ztrendcorrected)
        TailP = stats.chi2.sf(Fstat * df1, df1)
        r = stats.rankdata(Fstat, method="average")
        EmpiricalTailProb = (n - r + 0.5) / n
        ProbNotOutlier = np.minimum(TailP / EmpiricalTailProb, 1)
        df_pooled = n * df1
        df2_shrunk = np.full(n, df2)
        O = ProbNotOutlier < 1
        if np.any(O):
            df2_shrunk[O] = ProbNotOutlier[O] * df_pooled
            o = np.argsort(TailP)
            df2_shrunk[o] = np.maximum.accumulate(df2_shrunk[o])
        return {"scale": s20, "df2": df2, "tail.p.value": TailP, "df2.shrunk": df2_shrunk}

    def linkfun(val: float) -> float:
        return val / (1 + val)

    def linkinv(val: float) -> float:
        return val / (1 - val)

    def fun(val: float) -> float:
        df2_val = linkinv(val)
        mom = winsorized_moments(df1, df2_val)
        if trace:
            print("df2=", df2_val, ", Working Var=", mom["var"])
        return math.log(zwvar / mom["var"])

    if NonRobust["df2"] == math.inf:
        NonRobust["df2.shrunk"] = np.resize(NonRobust["df2"], n)
        return NonRobust

    rbx = linkfun(float(NonRobust["df2"]))
    funvalLow = fun(rbx)
    if funvalLow >= 0:
        df2 = float(NonRobust["df2"])
    else:
        root = brentq(fun, rbx, 1, xtol=1e-8, rtol=1e-8)
        df2 = linkinv(root)

    mom = winsorized_moments(df1, df2)
    ztrendcorrected = ztrend + zwmean - mom["mean"]
    s20 = np.exp(ztrendcorrected)
    zresid = z - ztrendcorrected
    Fstat = np.exp(zresid)
    LogTailP = _f_logsf(Fstat, df1, df2)
    TailP = np.exp(LogTailP)
    r = stats.rankdata(Fstat, method="average")
    LogEmpiricalTailProb = np.log(n - r + 0.5) - math.log(n)
    LogProbNotOutlier = np.minimum(LogTailP - LogEmpiricalTailProb, 0)
    ProbNotOutlier = np.exp(LogProbNotOutlier)
    ProbOutlier = -np.expm1(LogProbNotOutlier)
    if np.any(LogProbNotOutlier < 0):
        minLogTailP = float(np.min(LogTailP))
        if minLogTailP == -math.inf:
            df2_outlier = 0.0
            df2_shrunk = ProbNotOutlier * df2
        else:
            df2_outlier = math.log(0.5) / minLogTailP * df2
            new_log_tail = _f_logsf(np.max(Fstat), df1, df2_outlier)
            df2_outlier = math.log(0.5) / new_log_tail * df2_outlier
            df2_shrunk = ProbNotOutlier * df2 + ProbOutlier * df2_outlier
        o = np.argsort(LogTailP)
        df2_ordered = df2_shrunk[o]
        m_vals = np.cumsum(df2_ordered)
        m_vals = m_vals / (np.arange(n) + 1)
        imin = int(np.argmin(m_vals))
        df2_ordered[: imin + 1] = m_vals[imin]
        df2_shrunk[o] = np.maximum.accumulate(df2_ordered)
    else:
        df2_outlier = df2
        df2_shrunk = np.resize(df2, n)

    return {
        "scale": s20,
        "df2": df2,
        "tail.p.value": TailP,
        "prob.outlier": ProbOutlier,
        "df2.outlier": df2_outlier,
        "df2.shrunk": df2_shrunk,
    }


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
