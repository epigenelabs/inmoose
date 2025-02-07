# -----------------------------------------------------------------------------
# Copyright (C) 2023 M. Colange

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

"""
This module contains utilities to simulate RNASeq and single-cell RNASeq count
data. It follows the Splat model, as described in https://genomebiology.biomedcentral.com/articles/10.1186/s13059-017-1305-0.
Yet it is a completely independent implementation.
"""

import numpy as np
from scipy.special import expit
from scipy.stats import bernoulli, chi2, gamma, lognorm, poisson

from ..utils import Factor


def get_lognorm_factors(size, sel_prob, neg_prob, loc, scale, random_state):
    """
    Draw log-normal factors.

    Arguments
    ---------
    size : int or tuple of ints, optional
        Defining number of random variates
    sel_prob : float
        Probability that a factor is selected to be different of 1
    neg_prob : float
        Probability that a factor is less than 1
    loc : array_like, optional
        Location parameter of the log-normal distribution
    scale : array_like, optional
        Scale parameter of the log-normal distribution
    random_state : {int, `numpy.random.Generator` or `numpy.random.RandomState`},
                    optional
        If `random_state` is None (or `np.random`), the
        `numpy.random.RandomState` singleton is used.  If `random_state` is an
        int, a new ``RandomState`` instance is used, seeded with
        `random_state`.  If `random_state` is already a ``Generator`` or
        ``RandomState`` instance, that instance is used.

    Returns
    -------
    ndarray
        Random factors of given `size`
    """
    # make sure we have a Generator
    random_state = np.random.default_rng(random_state)

    selected = bernoulli.rvs(sel_prob, size=size, random_state=random_state) == 1
    n_selected = selected.sum()
    dir_selected = (-1) ** bernoulli.rvs(
        neg_prob, size=n_selected, random_state=random_state
    )
    # /!\ original R code uses `rlnorm(size, loc, scale)`, but the loc and scale correspond to the log-mean and log-sd
    #     therefore, the correct counterpart with scipy is `lognorm.rvs(s=scale, scale=np.exp(loc), size=size)`
    fac_selected = lognorm.rvs(
        s=scale, scale=np.exp(loc), size=n_selected, random_state=random_state
    )
    # reverse directions for factors that are less than one
    dir_selected[fac_selected < 1] *= -1
    factors = np.ones(size)
    factors[selected] = fac_selected**dir_selected
    return factors


def sim_rnaseq(
    nb_genes,
    nb_samples,
    batch=None,
    group=None,
    single_cell=False,
    alpha=0.6,
    beta=0.3,
    outlier_prob=0.05,
    outlier_location=4,
    outlier_scale=0.5,
    libsize_loc=11,
    libsize_scale=0.2,
    phi=0.1,
    bcv_df=60,
    x0=0,
    k=-1,
    random_state=None,
):
    """
    Simulate (sc)RNASeq data.

    For a precise description and understanding of the parameters, please refer
    to https://genomebiology.biomedcentral.com/articles/10.1186/s13059-017-1305-0.

    Arguments
    ---------
    nb_genes : int
        number of genes
    nb_samples : int
        number of samples
    batch : array-like, optional
        batch indices. Must have as many elements as `nb_samples`.
    group : array-like, optional
        vector/factor for biological condition of interest. Must have as many
        elements as `nb_samples`.
    single_cell : bool, optional
        if True, simulate scRNASeq data by adding drop-outs. Defaults to `False`.

    alpha : float, optional
        shape of the gamma distribution to draw the initial means from
    beta : float, optional
        rate of the gamme distribution to draw the initial means from
    outlier_prob : float, optional
        proportion of outliers
    outlier_location : float, optional
        location parameter for the log-normal distribution to draw the outlier
        inflation factors from
    outlier_scale : float, optional
        scale parameter for the log-normal distribution to draw the outlier
        inflation factors from
    libsize_loc : float, optional
        mean of the normal distribution to draw the log of the library sizes
        from
    libsize_scale : float, optional
        standard deviation of the normal distribution to draw the log of the
        library sizes from
    phi : float, optional
        common dispersion
    bcv_df : int, optional
        degrees of freedom in the inverse chi-square distribution used to draw
        the biological coefficient of variation parameters
    x0 : float, optional
        dropout midpoint, used to draw the dropouts. Unused if
        `single_cell=False`.
    k : float, optional
        dropout shape, used to draw the dropouts. Unused if `single_cell=False`.
    random_state : {int, `numpy.random.Generator` or `numpy.random.RandomState`},
                    optional
        If `random_state` is None (or `np.random`), the
        `numpy.random.RandomState` singleton is used.  If `random_state` is an
        int, a new ``RandomState`` instance is used, seeded with
        `random_state`.  If `random_state` is already a ``Generator`` or
        ``RandomState`` instance, that instance is used.

    Returns
    -------
    ndarray
        simulated count matrix, of size `nb_genes` x `nb_samples`
    """

    # make sure we have a Generator
    random_state = np.random.default_rng(random_state)

    if batch is None:
        batch = np.zeros(nb_samples, dtype=np.int64)
    batch = Factor(batch)
    assert len(batch) == nb_samples

    if group is None:
        group = np.zeros(nb_samples, dtype=np.int64)
    group = Factor(group)
    assert len(group) == nb_samples

    nb_batches = batch.nlevels()
    nb_groups = group.nlevels()

    # draw gene means
    original_means = gamma.rvs(
        alpha, scale=1 / beta, size=nb_genes, random_state=random_state
    )

    # draw outliers
    outlier_factor = get_lognorm_factors(
        original_means.shape,
        outlier_prob,
        0,
        outlier_location,
        outlier_scale,
        random_state,
    )
    is_outlier = outlier_factor != 1.0
    gene_means = np.where(
        is_outlier, outlier_factor * np.median(original_means), original_means
    )

    # draw library sizes
    L = lognorm.rvs(
        libsize_scale,
        scale=np.exp(libsize_loc),
        size=nb_samples,
        random_state=random_state,
    )

    # cell means
    cell_means = L * gene_means[:, None] / gene_means.sum()

    # biological coefficients of variation (squared)
    B2 = (phi + 1 / np.sqrt(cell_means)) ** 2 * (
        bcv_df / chi2.rvs(bcv_df, size=nb_genes, random_state=random_state)
    )[:, None]

    # trended cell means
    trended_cell_means = gamma.rvs(1 / B2, cell_means * B2, random_state=random_state)

    # DE fold-changes between cell types
    de_lf = [
        get_lognorm_factors(
            nb_genes, 0.1, 0.5, 0.3 + 0.2 * i, 0.1 + i * 0.5, random_state=random_state
        )
        for i in range(nb_groups)
    ]
    de_lf = np.choose(group.codes, [x[:, None] for x in de_lf])

    # cell type specific cell means
    de_trended_cell_means = de_lf * trended_cell_means

    # batch effect factors
    be_fact = [
        get_lognorm_factors(
            nb_genes, 1, 0.5, 0.1 + 0.5 * i, 1, random_state=random_state
        )
        for i in range(nb_batches)
    ]
    be_fact = np.choose(batch.codes, [x[:, None] for x in be_fact])
    batch_cell_means = be_fact * de_trended_cell_means

    # counts
    y = poisson.rvs(batch_cell_means, random_state=random_state)

    if single_cell:
        # simulate drop-outs
        pi = expit(k * (np.log(batch_cell_means) - x0))
        y = bernoulli.rvs(pi, random_state=random_state) * y

    return y
