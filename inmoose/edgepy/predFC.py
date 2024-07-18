# -----------------------------------------------------------------------------
# Copyright (C) 2008-2022 Yunshun Chen, Aaron TL Lun, Davis J McCarthy, Matthew E Ritchie, Belinda Phipson, Yifang Hu, Xiaobei Zhou, Mark D Robinson, Gordon K Smyth
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

# This file is based on the file 'R/predFC.R' of the Bioconductor edgeR package (version 3.38.4).


import numpy as np

from ..utils import LOGGER
from .addPriorCount import addPriorCount


def predFC_DGEList(
    self, design, prior_count=0.125, offset=None, dispersion=None, weights=None
):
    """
    Compute estimated coefficients for a negative binomial GLM in such a way
    that the log-fold-changes are shrunk towards zero.

    See also
    --------
    predFC

    Arguments
    ---------
    """
    if offset is None:
        offset = self.getOffset()
    if dispersion is None:
        dispersion = self.getDispersion()
    if dispersion is None:
        dispersion = 0
        LOGGER.warning("dispersion set to zero")

    return predFC(
        y=self.counts,
        design=design,
        prior_count=prior_count,
        offset=offset,
        dispersion=dispersion,
        weights=weights,
    )


def predFC(y, design, prior_count=0.125, offset=None, dispersion=0, weights=None):
    """
    Compute estimated coefficients for a negative binomial GLM in such a way
    that the log-fold-changes are shrunk towards zero.

    This function computes predictive log-fold changes (pfc) for a NB GLM. The
    pfc are posterior Bayesian estimators of the true log-fold-changes. They are
    predictive of values that might be replicated in a future experiment.

    Specifically, the function adds a small prior count to each observation
    before fitting the GLM (see :func:`addPriorCount` for details). The actual
    prior count that is added is proportional to the library size. This has the
    effect that any log-fold-change that was zero prior to augmentation remains
    zero and non-zero log-fold-changes are shrunk towards zero.

    The prior counts can be viewed as equivalent to a prior belief that the
    log-fold-changes are small, and the output can be viewed as posterior
    log-fold-changes from this Bayesian viewpoint. The output coefficients are
    called *predictive* log-fold-changes because, depending on the prior, They
    may be a better prediction of the true log-fold-changes than the raw
    estimates.

    Log-fold-changes for genes with low counts are shrunk more than those for
    genes with high counts. In particular, infinite log-fold-changes arising
    from zero counts are avoided. The exact degree to which this is done depends
    on the negative binomial dispersion.

    See also
    --------
    glmFit, exactTest, addPriorCount

    Arguments
    ---------
    y : array_like
        matrix of counts
    design : array_like
        the design matrix for the experiment
    prior_count : float
        the average prior count to be added to each observation. Larger values
        produce more shrinkage.
    offset : array_like
        vector or matrix giving the offset in the log-linear model predicto,
        as in :func:`glmFit`. Usually equal to log library size.
    dispersion : array_like
        vector of negative binomial dispersions
    weights : array_like, optional
        observation weights

    Returns
    -------
    ndarray
        matrix of (shrunk) linear model coefficients on the log2 scale

    References
    ----------
    B. Phipson. 2013. Empirical Bayes modelling of expression profiles and their
    associations. PhD thesis. University of Melbourne, Australia.
    http://repository.unimelb.edu.au/10187/17614
    """
    from .glmFit import glmFit

    # Add prior counts in proportion to library size
    (out_y, out_offset) = addPriorCount(y, offset=offset, prior_count=prior_count)

    # Check design
    design = np.asarray(design)

    # Return matrix of coefficients on log2 scale
    g = glmFit(
        out_y,
        design,
        offset=out_offset,
        dispersion=dispersion,
        prior_count=0,
        weights=weights,
    )
    return g.coefficients / np.log(2)
