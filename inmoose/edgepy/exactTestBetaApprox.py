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

# This file is based on the file 'R/exactTestBetaApprox.R' of the Bioconductor edgeR package (version 3.38.4).

import numpy as np
from scipy.stats import beta


def exactTestBetaApprox(y1, y2, dispersion=0.0):
    """
    Compute genewise *p*-values for differences in the means between two groups of negative-binomially distributed counts.

    This function implements an asymptotic beta distribution approximation to
    the conditional count distribution.

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

    # convert matrices to vectors
    ntags = y1.shape[0]
    n1 = y1.shape[1]
    n2 = y2.shape[1]
    y1 = np.round(y1.sum(axis=1))
    y2 = np.round(y2.sum(axis=1))

    dispersion = np.broadcast_to(dispersion, (ntags,))

    # Null fitted values
    y = y1 + y2
    mu = y / (n1 + n2)

    # Compute p-values
    pvals = np.ones(ntags)
    all_zero = y <= 0
    alpha1 = n1 * mu / (1 + dispersion * mu)
    alpha2 = n2 / n1 * alpha1
    med = np.zeros(ntags)
    med[~all_zero] = beta.ppf(0.5, alpha1[~all_zero], alpha2[~all_zero])
    left = ((y1 + 0.5) / y < med) & ~all_zero
    if left.any():
        pvals[left] = 2 * beta.cdf(
            (y1[left] + 0.5) / y[left], alpha1[left], alpha2[left]
        )
    right = ((y1 - 0.5) / y > med) & ~all_zero
    if right.any():
        pvals[right] = 2 * beta.sf(
            (y1[right] - 0.5) / y[right], alpha1[right], alpha2[right]
        )
    return pvals
