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

# This file is based on the file 'R/estimateGLMTagwiseDisp.R' of the Bioconductor edgeR package (version 3.38.4).


import numpy as np

from ..utils import LOGGER
from .aveLogCPM import aveLogCPM
from .dispCoxReidInterpolateTagwise import dispCoxReidInterpolateTagwise


def estimateGLMTagwiseDisp_DGEList(
    self, design=None, prior_df=10, trend=None, span=None
):
    """
    Compute an empirical Bayes estimate of the negative binomial dispersion
    parameter for each tag, with expression levels specified by a log-linear
    model.

    NB: Modifies :code:`self` in place

    Arguments
    ---------
    self : DGEList
    design : matrix
        the design matrix, as in :code:`glmFit`
    prior_df : int
        prior degrees of freedom
    trend : bool
        whether the prior should be the trended (:code:`trend=True`) or the
        common dispersion (:code:`trend=False`)
    span : float, optional
        width of the smoothing window, in terms of proportion of the data set.
        Default value decreases with the number of tags.

    Returns
    -------
    DGEList
        the input :code:`self`, updated with the tagwise dispersion parameter
        estimate for each tag for the negative binomial model that maximizes the
        Cox-Reid adjusted profile likelihood.
    """
    # Find appropriate dispersion
    if trend is None:
        trend = self.trended_dispersion is not None
    if trend:
        dispersion = self.trended_dispersion
        if dispersion is None:
            raise ValueError(
                "No trended_dispersion in data object. Run estimateGLMTrendedDisp first."
            )
    else:
        dispersion = self.common_dispersion
        if dispersion is None:
            raise ValueError(
                "No common_dispersion found in data object. Run estimateGLMCommonDisp first."
            )

    if self.AveLogCPM is None:
        self.AveLogCPM = self.aveLogCPM()
    ntags = self.counts.shape[0]

    if span is None:
        if ntags > 10:
            span = (10 / ntags) ** 0.23
        else:
            span = 1
    self.span = span

    d = estimateGLMTagwiseDisp(
        self.counts,
        design=design,
        offset=self.getOffset(),
        dispersion=dispersion,
        trend=trend,
        span=span,
        prior_df=prior_df,
        AveLogCPM=self.AveLogCPM,
        weights=self.weights,
    )
    self.prior_df = prior_df
    self.tagwise_dispersion = d
    return self


def estimateGLMTagwiseDisp(
    y,
    dispersion,
    design=None,
    offset=None,
    prior_df=10,
    trend=True,
    span=None,
    AveLogCPM=None,
    weights=None,
):
    """
    Compute an empirical Bayes estimate of the negative binomial dispersion
    parameter for each tag, with expression levels specified by a log-linear
    model.

    This function implements the empirical Bayes strategy propose by McCarthy et
    al. (2012) [2]_ for estimating the tagwise negative binomial dispersions.
    The experimental conditions are specified by design matrix allowing for
    multiple explanatory factors. The empirical Bayes posterior is implemented
    as a conditional likelihood with tag-specific weights, and the conditional
    likelihood is computed using Cox-Reid approximate conditional likelihood
    (Cox and Reid, 1987 [1]_).

    The prior degrees of freedom determine the weight given to the global
    dispersion trend. The larger the prior degrees of freedom, the more the
    tagwise dispersions are squeezed towards the global trend.

    Note that the terms "tag" and "gene" are synonymous here. The function is
    only named "tagwise" for historical reasons.

    This function calls the lower-level :func:`dispCoxReidInterpolateTagwise`.

    Arguments
    ---------
    y : matrix
        matrix of counts
    dispersion : float or array_like
        common or trended dispersion estimates, used as an initial estimate for
        the tagwise estimates
    design : matrix
        the design matrix, as in :code:`glmFit`
    offset : matrix, optional
        offset matrix for the log-linear model, as in :func:`glmFit`. Defaults
        to the log-effective library sizes.
    prior_df : int
        prior degrees of freedom
    trend : bool
        whether the prior should be the trended (:code:`trend=True`) or the
        common dispersion (:code:`trend=False`)
    span : float, optional
        width of the smoothing window, in terms of proportion of the data set.
        Default value decreases with the number of tags.
    AveLogCPM : array_like, optional
        vector of the average log2 counts per million for each tag
    weights : matrix, optional
        observation weights

    Returns
    -------
    ndarray
        vector of the tagwise dispersion estimates

    References
    ----------
    .. [1] D. R. Cox, N. Reid. 1987. Parameter orthogonality and approximate
       conditional inference. Journal of the Royal Statistical Society Series B
       49, 1-39.
    .. [2] D. J. McCarthy, Y. Chen, G. K. Smyth. 2012. Differential expression
       analysis of multifactor RNA-Seq experiments with respect to biological
       variation. Nucleic Acids Research 40, 4288-4297. :doi:`10.1093/nar/gks042`
    """

    # Check y
    y = np.asarray(y)
    (ntags, nlibs) = y.shape
    if ntags == 0:
        return 0

    # Check design
    if design is None:
        design = np.ones((y.shape[1], 1))
    else:
        design = np.asarray(design)

    if design.shape[1] >= y.shape[1]:
        LOGGER.warning("No residual df: setting dispersion to NA")
        return np.full((ntags,), np.nan)

    # Check offset
    if offset is None:
        offset = np.log(y.sum(axis=0))

    # Check span
    # span can be chosen smaller when ntags is large
    if span is None:
        if ntags > 10:
            span = (10 / ntags) ** 0.23
        else:
            span = 1

    # Check AveLogCPM
    if AveLogCPM is None:
        AveLogCPM = aveLogCPM(y, offset=offset, weights=weights)

    # Call Cox-Reid grid method
    tagwise_dispersion = dispCoxReidInterpolateTagwise(
        y,
        design,
        offset=offset,
        dispersion=dispersion,
        trend=trend,
        prior_df=prior_df,
        span=span,
        AveLogCPM=AveLogCPM,
        weights=weights,
    )
    return tagwise_dispersion
