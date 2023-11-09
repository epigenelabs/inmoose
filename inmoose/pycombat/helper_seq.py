# -----------------------------------------------------------------------------
# Copyright (C) 2019-2020 Yuqing Zhang
# Copyright (C) 2022-2023 Maximilien Colange

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

# This file is based on the file 'R/helper_seq.R' of the Bioconductor sva package (version 3.44.0).


import numpy as np
from scipy.stats import nbinom


def vec2mat(vec, n_times):
    """
    Expand a vector into matrix (columns as the original vector)
    """
    vec = np.asarray(vec)
    vec = vec.reshape(vec.shape[0], 1)
    return np.full((vec.shape[0], n_times), vec)


def match_quantiles(counts_sub, old_mu, old_phi, new_mu, new_phi):
    """
    Match quantiles from a source negative binomial distribution to a target
    negative binomial distribution.

    Arguments
    ---------
    counts_sub : array_like
        the original data following the source distribution
    old_mu : array_like
        the mean of the source distribution
    old_phi : array_like
        the dispersion of the source distribution
    new_mu : array_like
        the mean of the target distribution
    new_phi : array_like
        the dispersion of the target distribution

    Returns
    -------
    ndarray
        adjusted data, corresponding in the target distribution to the same
        quantiles as the input data in the source distribution
    """
    new_counts_sub = np.full(counts_sub.shape, np.nan)

    i = counts_sub <= 1
    new_counts_sub[i] = counts_sub[i]
    i = np.logical_not(i)
    old_size = np.full(old_mu.shape, 1 / old_phi.reshape(old_mu.shape[0], 1))
    new_size = np.full(new_mu.shape, 1 / new_phi.reshape(new_mu.shape[0], 1))
    old_prob = old_size / (old_size + old_mu)
    new_prob = new_size / (new_size + new_mu)
    tmp_p = nbinom.cdf(counts_sub[i] - 1, old_size[i], old_prob[i])
    new_counts_sub[i] = np.where(
        np.abs(tmp_p - 1) < 1e-4,
        counts_sub[i],
        1 + nbinom.ppf(tmp_p, new_size[i], new_prob[i]),
    )

    # Original (pythonized) R code for reference
    #
    # for a in range(counts_sub.shape[0]):
    #    for b in range(counts_sub.shape[1]):
    #        if counts_sub[a,b] <= 1:
    #            new_counts_sub[a,b] = counts_sub[a,b]
    #        else:
    #            tmp_p = pnbinom_opt(counts_sub[a,b]-1, mu=old_mu[a,b], size=1/old_phi[a])
    #            if abs(tmp_p-1) < 1e-4:
    #                # for outlier count, if tmp_p==1, qnbinom(tmp_p) will return Inf values -> use original count instead
    #                new_counts_sub[a,b] = counts_sub[a,b]
    #            else:
    #                new_counts_sub[a,b] = 1+qnbinom_opt(tmp_p, mu=new_mu[a,b], size=1/new_phi[a])

    return new_counts_sub
