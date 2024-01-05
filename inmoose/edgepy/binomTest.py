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

# This file is based on the file 'R/binomTest.R' of the Bioconductor edgeR package (version 3.38.4).

import numpy as np
from scipy.stats import binom, chi2_contingency


def binomTest(y1, y2, n1=None, n2=None, p=None):
    """
    Computes *p*-values for differential abundance for each gene between two digital libraries, conditioning on the total counts for each gene. The counts in each group as a proportion of the whole are assumed to follow a binomial distribution.

    This function can be used to compare two libraries from SAGE, RAN-Seq,
    ChIP-Seq or other sequencing technologies with respect to technical
    variation.

    An exact two-sided binomial test is computed for each gene. This test is
    closely releated to Fisher's exact test for 2x2 contingency tables but,
    unlike Fisher's test, it conditions on the total number of counts for each
    gene. The null hypothesis is that the expected counts are in the same
    proportions as the library sizes, *i.e.* that the binomial probability for
    the first library is :code:`n1 / (n1+n2)`.

    The two-sided rejection region is chosen analogously to Fisher's test.
    Specifically, the rejection region consists of those values with smallest
    probabilities under the null hypothesis.

    Then the counts are reasonably large, the binomial test, Fisher's test and
    Pearson's chi square all give the same results. When the counts are
    smaller, the binomial test is usually to be preferred in this context.

    References
    ----------
    `Binomial test <http://en.wikipedia.org/wiki/Binomial_test>`_
    `Fisher's exact test <http://en.wikipedia.org/wiki/Fisher's_exact_test>`_
    `Serial analysis of gene expression <http://en.wikipedia.org/wiki/Serial_analysis_of_gene_expression>`_

    Arguments
    ---------
    y1 : array_like
        integer array giving the count for each gene in the first library.
        Non-integer values are rounded to the nearest integer.
    y2 : array_like
        integer array giving the count for each gene in the second library.
        Non-integer values are rounded to the nearest integer.
    n1 : int
        total number of counts in the first library, across all genes. Not
        required if :code:`p` is supplied.
    n2 : int
        total number of counts in the second library, across all genes. Not
        required if :code:`p` is supplied.
    p : float
        expected proportion of :code:`y1` to the total for each gene under the
        null hypothesis.

    Returns
    -------
    ndarray
        array of *p*-values
    """
    if np.isnan(y1).any() or np.isnan(y2).any():
        raise ValueError("missing values not allowed")
    y1 = np.round(y1)
    y2 = np.round(y2)

    if y1.ndim > 1 or y2.ndim > 1:
        raise ValueError("y1 and y2 must be 1-D arrays")
    if len(y1) != len(y2):
        raise ValueError("y1 and y2 must have same length")
    if (y1 < 0).any() or (y2 < 0).any():
        raise ValueError("y1 and y2 must be non-negative")

    if n1 is None:
        n1 = y1.sum()
    if n2 is None:
        n2 = y2.sum()
    if p is None:
        p = n1 / (n1 + n2)

    if p <= 0 or p >= 1:
        raise ValueError("p must be between 0 and 1")
    size = y1 + y2
    p_value = np.ones(len(y1))
    if p == 0.5:
        i = size > 0
        if i.any():
            y1 = np.minimum(y1[i], y2[i])
            size = size[i]
            p_value[i] = np.minimum(2 * binom.cdf(y1, n=size, p=0.5), 1)
        return p_value
    big = size > 10000
    if big.any():
        for i in range(len(y1)):
            if big[i]:
                p_value[i] = chi2_contingency(
                    np.array([[y1[i], n1 - y1[i]], [y2[i], n2 - y2[i]]])
                ).pvalue
    size0 = size[(size > 0) & ~big]
    for isize in np.unique(size0):
        i = size == isize
        d = binom.pmf(np.arange(isize + 1), p=p, n=isize)
        o = np.argsort(d, kind="stable")
        cumsump = d[o].cumsum()[np.argsort(o, kind="stable")]
        p_value[i] = cumsump[y1[i]]
    return p_value
