# -----------------------------------------------------------------------------
# Copyright (C) 2013-2022 Michael I. Love, Constantin Ahlmann-Eltze
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

# This file is based on the file 'R/core.R' of the Bioconductor DESeq2 package
# (version 3.16).


import numpy as np

from ..utils import Factor


def wtd_quantile(
    x, weights=None, probs=[0, 0.25, 0.5, 0.75, 1], normwt=False, na_rm=True
):
    """
    compute weighted quantiles

    This function is copied from R package `Hmisc
    <https://hbiostat.org/r/hmisc/>`_, to avoid extra package dependencies,
    with the alteration of leaving out parameter :code:`type` (not used in
    DESeq) and commenting out the :code:`isdate` test.

    Note that this copy was already done in the original DESeq2 R code.
    """
    if np.any((probs < 0) | (probs > 1)):
        raise ValueError("Probabilities must be between 0 and 1 inclusive")

    if weights is None:
        if na_rm:
            return np.nanquantile(x, q=probs)
        else:
            return np.quantile(x, q=probs)

    i = np.isnan(weights) | weights == 0
    if np.any(i):
        x = x[~i]
        weights = weights[~i]

    (x, wts) = wtd_table(x, weights, na_rm=na_rm, normwt=normwt)
    n = np.sum(wts)
    order = 1 + (n - 1) * probs
    low = np.maximum(np.floor(order), 1)
    high = np.minimum(low + 1, n)
    order = order % 1
    ## Find low and high order statistics
    ## These are the minimum values of x such that the cum. freqs >= (low,high)
    allq = np.piecewise(
        np.hstack([low, high]),
        True,
        [lambda v: np.hstack([x, [x[-1]]])[np.searchsorted(np.cumsum(wts), v)]],
    )
    try:
        k = len(probs)
    except TypeError:
        k = 1
    quantiles = (1 - order) * allq[:k] + order * allq[-k]
    return quantiles


def wtd_table(x, weights=None, normwt=False, na_rm=True):
    if weights is None:
        weights = np.ones(len(x))
    else:
        weights = np.asarray(weights)

    x = Factor(x)
    lev = x.categories
    x = x.codes

    if na_rm:
        s = ~np.isnan(x + weights)
        x = x[s]
        weights = weights[s]

    if normwt:
        weights = weights * len(x) / np.sum(weights)

    i = np.argsort(x)
    x = x[i]
    weights = weights[i]

    if len(np.unique(x)) != len(x):
        x = np.asarray(x)
        weights = np.array([np.sum(weights[x == xx]) for xx in np.unique(x)])
        if len(lev) > 0:
            levused = lev[np.sort(np.unique(x))]
            if len(weights) > len(levused) and np.any(np.isnan(weights)):
                weights = weights[~np.isnan(weights)]

            if len(weights) != len(levused):
                raise RuntimeError("program logic error")

        return (levused, weights)

    if len(lev) > 0:
        return (lev[x], weights)
    else:
        return (x, weights)
