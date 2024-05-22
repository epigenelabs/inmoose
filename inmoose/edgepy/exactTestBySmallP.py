# -----------------------------------------------------------------------------
# Copyright (C) 2008-2022 Yunshun Chen, Aaron TL Lun, Davis J McCarthy, Matthew E Ritchie, Belinda Phipson, Yifang Hu, Xiaobei Zhou, Mark D Robinson, Gordon K Smyth
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

# This file is based on the file 'R/exactTestBySmallP.R' of the Bioconductor edgeR package (version 3.38.4).

import numpy as np

from ..utils import dnbinom_mu as dnbinom
from .binomTest import binomTest
from .exactTestDoubleTail import exactTestDoubleTail


def exactTestBySmallP(y1, y2, dispersion=0):
    """
    Compute genewise *p*-values for differences in the means between two groups of negative-binomially distributed counts.

    This function implements the method of small probabilities as proposed by
    [Robinson2008]_. This method corresponds exactly to :func:`binomTest` as the
    dispersion approaches zero, but gives poor results when the dispersion is
    very large.

    See also
    --------
    exactTest

    Arguments
    ---------
    y1 : matrix
        matrix of counts for the first of the two experimental groups to be
        tested for differences. Rows correspond to genes and columns to
        libraries. Libraries are assumed to be equal in size -- *e.g.* adjusted
        pseudocounts from the output of :func:`equalizeLibSizes`.
    y2 : matrix
        matrix of counts for the second of the two experimental groups to be
        tested for differences. Rows correspond to genes and columns to
        libraries. Libraries are assumed to be equal in size -- *e.g.* adjusted
        pseudocounts from the output of :func:`equalizeLibSizes`.
    dispersion : array_like of floats
        an array of dispersions, either of length one or of length equal to the
        number of genes.

    Returns
    -------
    ndarray
        array of genewise *p*-values, one for each row of :code:`y1` and :code:`y2`
    """
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    if y1.shape[0] != y2.shape[0]:
        raise ValueError("Number of rows of y1 not equal to number of rows of y2")
    ntags = y1.shape[0]
    if np.isnan(y1).any() or np.isnan(y2).any():
        raise ValueError("NAs not allowed")
    n1 = y1.shape[1]
    n2 = y2.shape[1]

    if n1 == n2:
        return exactTestDoubleTail(y1=y1, y2=y2, dispersion=dispersion)

    dispersion = np.asarray(dispersion)
    sum1 = np.round(y1.sum(axis=1))
    sum2 = np.round(y2.sum(axis=1))
    if (dispersion == 0).all():
        return binomTest(sum1, sum2, p=n1 / (n1 + n2))
    if (dispersion == 0).any():
        raise ValueError("dispersion must be either all zero or all positive")
    dispersion = np.broadcast_to(dispersion, (ntags,))
    N = sum1 + sum2
    mu = N / (n1 + n2)
    r = 1 / dispersion
    all_zeros = N == 0

    pvals = np.ones(ntags)
    if ntags == 0:
        return pvals
    if any(all_zeros):
        pvals[~all_zeros] = exactTestBySmallP(
            y1=y1[~all_zeros, :],
            y2=y2[~all_zeros, :],
            dispersion=dispersion[~all_zeros],
        )
        return pvals

    for i in range(ntags):
        ind = np.arange(N[i] + 1)
        p_top = dnbinom(ind, size=n1 * r[i], mu=n1 * mu[i]) * dnbinom(
            N[i] - ind, size=n2 * r[i], mu=n2 * mu[i]
        )
        p_obs = dnbinom(sum1[i], size=n1 * r[i], mu=n1 * mu[i]) * dnbinom(
            sum2[i], size=n2 * r[i], mu=n2 * mu[i]
        )
        keep = p_top <= p_obs
        p_bot = dnbinom(N[i], size=(n1 + n2) * r[i], mu=(n1 + n2) * mu[i])
        pvals[i] = np.sum(p_top[keep] / p_bot)

    # edgeR code returns "min(pvals, 1)" but it looks like a typo: "pmin(pvals, 1)"
    # anyhow, we choose to align with edgeR behavior here
    # probably not a big deal as "smallp" is no longer recommended
    return np.min([np.min(pvals), 1])
