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

# This file is based on the file 'R/exactTestByDeviance.R' of the Bioconductor edgeR package (version 3.38.4).

import numpy as np
from scipy.stats import nbinom

from .binomTest import binomTest
from .edgepy_cpp import compute_unit_nb_deviance
from .exactTestDoubleTail import exactTestDoubleTail


def exactTestByDeviance(y1, y2, dispersion=0.0):
    """
    Compute genewise *p*-values for differences in the means between two groups of negative-binomially distributed counts.

    This function uses the deviance goodness of fit statistics to define the
    rejection region, and is therefore equivalent to a conditional likelihood
    ratio test.

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

    pvals = np.zeros(ntags)
    if ntags == 0:
        return pvals

    # Eliminate all zero rows
    all_zeros = (sum1 == 0) & (sum2 == 0)
    if all_zeros.any():
        pvals[~all_zeros] = exactTestByDeviance(
            y1=y1[~all_zeros, :],
            y2=y2[~all_zeros, :],
            dispersion=dispersion[~all_zeros],
        )
        pvals[all_zeros] = 1
        return pvals

    # The code below was originally written in C++
    nlibs = n1 + n2
    stotal = sum1 + sum2
    mu = stotal / nlibs
    mu1 = mu * n1
    mu2 = mu * n2
    r1 = n1 / dispersion
    r2 = n2 / dispersion
    p = r1 / (r1 + mu1)

    # The aim is to sum conditional probabilities for all partitions of the
    # total sum with deviances greater than that observed for the current
    # partition. We start computing from the extremes in both cases
    phi1 = 1 / r1
    phi2 = 1 / r2

    for i in range(ntags):
        obsdev = compute_unit_nb_deviance(
            sum1[i], mu1[i], phi1[i]
        ) + compute_unit_nb_deviance(sum2[i], mu2[i], phi2[i])

        # Going from the left
        for j in range(int(stotal[i]) + 1):
            if obsdev <= compute_unit_nb_deviance(
                j, mu1[i], phi1[i]
            ) + compute_unit_nb_deviance(stotal[i] - j, mu2[i], phi2[i]):
                pvals[i] += nbinom.pmf(j, r1[i], p[i]) * nbinom.pmf(
                    stotal[i] - j, r2[i], p[i]
                )
            else:
                break

        # Going from the right, or what's left of it
        for k in range(int(stotal[i]) - j + 1):
            if obsdev <= compute_unit_nb_deviance(
                k, mu2[i], phi2[i]
            ) + compute_unit_nb_deviance(stotal[i] - k, mu1[i], phi1[i]):
                pvals[i] += nbinom.pmf(k, r2[i], p[i]) * nbinom.pmf(
                    stotal[i] - k, r1[i], p[i]
                )
            else:
                break

    totalr = r1 + r2
    pvals /= nbinom.pmf(stotal, totalr, totalr / (totalr + mu1 + mu2))

    return np.minimum(pvals, 1)
