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

# This file is based on the file 'R/exactTestDoubleTail.R' of the Bioconductor edgeR package (version 3.38.4).

import numpy as np

from ..utils import dnbinom_mu as dnbinom
from .binomTest import binomTest
from .exactTestBetaApprox import exactTestBetaApprox


def exactTestDoubleTail(y1, y2, dispersion=0, big_count=900):
    """
    Compute genewise *p*-values for differences in the means between two groups of negative-binomially distributed counts.

    This function computes two-sided *p*-values by doubling the smaller tail
    probability.

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
    big_count : int
        count size above which asymptotic beta approximation will be used.

    Returns
    -------
    ndarray
        array of genewise *p*-values, one for each row of :code:`y1` and :code:`y2`
    """
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)

    # convert matrices to vectors
    ntags = y1.shape[0]
    n1 = y1.shape[1]
    n2 = y2.shape[1]
    s1 = np.round(y1.sum(axis=1))
    s2 = np.round(y2.sum(axis=1))

    dispersion = np.broadcast_to(dispersion, (ntags,))

    # Null fitted values
    s = s1 + s2
    mu = s / (n1 + n2)
    mu1 = n1 * mu
    mu2 = n2 * mu

    pvals = np.ones(ntags)

    # Poisson case
    pois = dispersion <= 0
    # binomTest does not use equal tailed rejection region
    if pois.any():
        pvals[pois] = binomTest(s1[pois], s2[pois], p=n1 / (n1 + n2))

    # Use beta approximation for large counts
    big = (s1 > big_count) & (s2 > big_count)
    if big.any():
        pvals[big] = exactTestBetaApprox(y1[big, :], y2[big, :], dispersion[big])

    pbot = np.zeros(ntags)
    size1 = np.zeros(ntags)
    size2 = np.zeros(ntags)
    left = (s1 < mu1) & ~pois & ~big
    if left.any():
        pbot[left] = dnbinom(s[left], size=(n1 + n2) / dispersion[left], mu=s[left])
        size1[left] = n1 / dispersion[left]
        size2[left] = n2 / dispersion[left]
        for g in np.nonzero(left)[0]:
            x = np.arange(s1[g] + 1)
            p_top = dnbinom(x, size=size1[g], mu=mu1[g]) * dnbinom(
                s[g] - x, size=size2[g], mu=mu2[g]
            )
            pvals[g] = 2 * p_top.sum()
        pvals[left] = pvals[left] / pbot[left]

    right = (s1 > mu1) & ~pois & ~big
    if right.any():
        pbot[right] = dnbinom(s[right], size=(n1 + n2) / dispersion[right], mu=s[right])
        size1[right] = n1 / dispersion[right]
        size2[right] = n2 / dispersion[right]
        for g in np.nonzero(right)[0]:
            x = np.arange(s1[g], s[g] + 1)
            p_top = dnbinom(x, size=size1[g], mu=mu1[g]) * dnbinom(
                s[g] - x, size=size2[g], mu=mu2[g]
            )
            pvals[g] = 2 * p_top.sum()
        pvals[right] = pvals[right] / pbot[right]

    return np.minimum(pvals, 1)
