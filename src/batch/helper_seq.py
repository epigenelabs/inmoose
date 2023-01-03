import numpy as np
from scipy.stats import nbinom

def vec2mat(vec, n_times):
    """
    Expand a vector into matrix (columns as the original vector)
    """
    vec = np.asarray(vec)
    vec = vec.reshape(vec.shape[0],1)
    return np.full((vec.shape[0], n_times), vec)

def match_quantiles(counts_sub, old_mu, old_phi, new_mu, new_phi):
    """
    Match quantiles from a source negative binomial distribution to a target negative binomial distribution.

    :param counts_sub: the original data following the source distribution
    :type counts_sub: matrix
    :param old_mu: the mean of the source distribution
    :type old_mu: matrix
    :param old_phi: the dispersion of the source distribution
    :type old_phi: vector
    :param new_mu: the mean of the target distribution
    :type new_mu: matrix
    :param new_phi: the dispersion of the target distribution
    :type new_phi: vector

    :return: adjusted data, corresponding in the target distribution to the same quantiles as the input data has in the source distribution
    :rtype: matrix
    """
    new_counts_sub = np.full(counts_sub.shape, np.nan)

    i = counts_sub <= 1
    new_counts_sub[i] = counts_sub[i]
    i = np.logical_not(i)
    old_size = np.full(old_mu.shape, 1 / old_phi.reshape(old_mu.shape[0], 1))
    new_size = np.full(new_mu.shape, 1 / new_phi.reshape(new_mu.shape[0], 1))
    old_prob = old_size / (old_size + old_mu)
    new_prob = new_size / (new_size + new_mu)
    tmp_p = nbinom.cdf(counts_sub[i]-1, old_size[i], old_prob[i])
    new_counts_sub[i] = np.where(np.abs(tmp_p-1) < 1e-4, counts_sub[i], 1+nbinom.ppf(tmp_p, new_size[i], new_prob[i]))

    # Original (pythonized) R code for reference
    #
    #for a in range(counts_sub.shape[0]):
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

