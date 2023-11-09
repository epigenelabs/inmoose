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

# This file is based on the file 'R/glmfit.R' of the Bioconductor edgeR package (version 3.38.4).


import numpy as np

from ..utils import asfactor
from .DGEGLM import DGEGLM
from .makeCompressedMatrix import _compressDispersions, _compressOffsets
from .mglmLevenberg import mglmLevenberg
from .mglmOneWay import designAsFactor, mglmOneWay
from .nbinomDeviance import nbinomDeviance
from .predFC import predFC

from patsy import dmatrix


def glmFit_DGEList(self, design=None, dispersion=None, prior_count=0.125, start=None):
    """
    Fit a negative binomial generalized log-linear model to the read counts for
    each gene. Conduct genewise statistical tests for a given coefficient or
    coefficient contrast.

    See also
    --------
    glmFit

    Arguments
    ---------
    design : matrix, optional
        design matrix for the genewise linear models. Must be of full column
        rank. Defaults to a single column of ones, equivalent to treating the
        columns as replicate libraries.
    dispersion : float or array_like
        scalar, vector or matrix of negative binomial dispersions. Can be a
        common value for all genese, a vector of dispersion values with one for
        each gene, or a matrix of dispersion values with one for each observation.
        If :code:`None`, it will be extracted from :code:`y`, with order of
        precedence: genewise dispersion, trended dispersion, common dispersion.
    prior_count : float
        average prior count to be added to observation to shrink the estimated
        log-fold-change towards zero.
    start : matrix, optional
        initial estimates for the linear model coefficients

    Returns
    -------
    DGEList
        object containing the data about the fit
    """

    # The design matrix defaults to the oneway layout defined by self.samples.group
    # If there is only one group, then the design matrix is left None so that a matrix with a single intercept column will be set later by glmFit.
    if design is None:
        design = self.design
        if design is None:
            group = asfactor(self.samples.group).droplevels()
            if group.nlevels() > 1:
                design = dmatrix("~C(self.samples.group)")

    if dispersion is None:
        dispersion = self.getDispersion()
    if dispersion is None:
        raise ValueError("No dispersion values found in DGEList object")
    offset = self.getOffset()
    if self.AveLogCPM is None:
        self.AveLogCPM = self.aveLogCPM()

    fit = glmFit(
        y=self.counts,
        design=design,
        dispersion=dispersion,
        offset=offset,
        lib_size=None,
        weights=self.weights,
        prior_count=prior_count,
        start=start,
    )

    fit.samples = self.samples
    fit.genes = self.genes
    fit.prior_df = self.prior_df
    fit.AveLogCPM = self.AveLogCPM
    return fit


def glmFit(
    y,
    design=None,
    dispersion=None,
    offset=None,
    lib_size=None,
    weights=None,
    prior_count=0.125,
    start=None,
):
    """
    Fit a negative binomial generalized log-linear model to the read counts for
    each gene. Conduct genewise statistical tests for a given coefficient or
    coefficient contrast.

    This function implements one of the GLM methods developed by McCarthy et al.
    (2012) [1]_.

    :code:`glmFit` fits genewise negative binomial GLMs, all with the same
    design matrix but possibly different dispersions, offsets and weights.
    When the design matrix defines a one-way layout, or can be re-parameterized
    to a one-way layout, the GLMs are fitting very quickly using
    :func:`mglmOneGroup`. Otherwise the default fitting method, implemented in
    :func:`mglmLevenberg`, uses a Fisher scoring algorithm with Levenberg-style
    damping.

    Positive :code:`prior_count` cause the returned coefficients to be shrunk in
    such a way that fold-changes between the treatment conditions are decreased.
    In particular, infinite fold-changes are avoided. Larger values cause more
    shrinkage. The returned coefficients are affected but not the likelihood
    ratio tests or p-values.

    See also
    --------
    mglmOneGroup : low-level computations
    mglmLevenberg : low-level computations

    Arguments
    ---------
    y : matrix
        matrix of counts
    design : matrix, optional
        design matrix for the genewise linear models. Must be of full column
        rank. Defaults to a single column of ones, equivalent to treating the
        columns as replicate libraries.
    dispersion : float or array_like
        scalar, vector or matrix of negative binomial dispersions. Can be a
        common value for all genese, a vector of dispersion values with one for
        each gene, or a matrix of dispersion values with one for each
        observation.
    offset : float or array_like, optional
        matrix of the same shape as :code:`y` giving offsets for the log-linear
        models. Can be a scalar or a vector of length :code:`y.shape[1]`, in
        which case it is broadcasted to the shape of :code:`y`.
    lib_size : array_like, optional
        vector of length :code:`y.shape[1]` giving library sizes. Only used if
        :code:`offset=None`, in which case :code:`offset` is set to
        :code:`log(lib_size)`. Defaults to :code:`colSums(y)`.
    weights : matrix, optional
        prior weights for the observations (for each library and gene) to be
        used in the GLM calculations
    prior_count : float
        average prior count to be added to observation to shrink the estimated
        log-fold-change towards zero.
    start : matrix, optional
        initial estimates for the linear model coefficients

    Returns
    -------
    DGEGLM
        object containing:

        - :code:`counts`, the input matrix of counts
        - :code:`design`, the input design matrix
        - :code:`weights`, the input weights matrix
        - :code:`offset`, matrix of linear model offsets
        - :code:`dispersion`, vector of dispersions used for the fit
        - :code:`coefficients`, matrix of estimated coefficients from the GLM
          fits, on the natural log scale, of size :code:`y.shape[0]` by
          :code:`design.shape[1]`.
        - :code:`unshrunk_coefficients`, matrix of estimated coefficients from
          the GLM fits when no log-fold-changes shrinkage is applied, on the
          natural log scale, of size :code:`y.shape[0]` by
          :code:`design.shape[1]`. It exists only when :code:`prior_count` is
          not 0.
        - :code:`fitted_values`, matrix of fitted values from GLM fits, same
          shape as :code:`y`
        - :code:`deviance`, numeric vector of deviances, one for each gene

    References
    ----------
    .. [1] D. J. McCarthy, Y. Chen, G. K. Smyth. 2012. Differential expression
       analysis of multifactor RNA-Seq experiments with respect to biological
       variation. Nucleic Acids Research 40, 4288-4297. :doi:`10.1093/nar/gks042`
    """
    # Check y
    y = np.asarray(y, order="F")
    (ntag, nlib) = y.shape

    # Check design
    if design is None:
        design = np.ones(shape=(nlib, 1), order="F")
    else:
        design = np.asarray(design, order="F")
        if design.shape[0] != nlib:
            raise ValueError("design should have as many rows as y has columns")
        if np.linalg.matrix_rank(design) < design.shape[1]:
            raise ValueError(
                "Design matrix is not full rank. Some coefficients are not estimable"
            )

    # Check dispersion
    if dispersion is None:
        raise ValueError("No dispersion values provided")
    dispersion = np.asanyarray(dispersion, order="F")
    # TODO check dispersion for NaN and non-numeric values
    if dispersion.shape not in [(), (1,), (ntag,), y.shape]:
        raise ValueError("Dimensions of dispersion do not agree with dimensions of y")
    dispersion_mat = _compressDispersions(y, dispersion)

    # Check offset
    if offset is not None:
        # TODO check that offset is numeric
        offset = np.asanyarray(offset, order="F")
        if offset.shape not in [(), (1,), (nlib,), y.shape]:
            raise ValueError("Dimensions of offset do not agree with dimensions of y")

    # Check lib_size
    if lib_size is not None:
        # TODO check that lib_size is numeric
        lib_size = np.asarray(lib_size, order="F")
        if lib_size.shape not in [(), (1,), (nlib,)]:
            raise ValueError("lib_size has wrong length, should agree with ncol(y)")

    # Consolidate lib_size and offset into a compressed matrix
    offset = _compressOffsets(y=y, lib_size=lib_size, offset=offset)

    # weights are checked in lower-level functions

    # Fit the tagwise GLMs
    # If the design is equivalent to a oneway layout, use a shortcut algorithm
    group = designAsFactor(design)
    if group.nlevels() == design.shape[1]:
        (coef, fitted_values) = mglmOneWay(
            y,
            design=design,
            group=group,
            dispersion=dispersion_mat,
            offset=offset,
            weights=weights,
            coef_start=start,
        )
        deviance = nbinomDeviance(
            y=y, mean=fitted_values, dispersion=dispersion_mat, weights=weights
        )
        fit_method = "oneway"
        fit = (coef, fitted_values, deviance, None, None)
    else:
        fit = mglmLevenberg(
            y,
            design=design,
            dispersion=dispersion_mat,
            offset=offset,
            weights=weights,
            coef_start=start,
            maxit=250,
        )
        fit_method = "levenberg"

    # Prepare output
    fit = DGEGLM(fit)
    fit.counts = y
    fit.method = fit_method
    if prior_count > 0:
        fit.unshrunk_coefficients = fit.coefficients
        fit.coefficients = predFC(
            y,
            design,
            offset=offset,
            dispersion=dispersion_mat,
            prior_count=prior_count,
            weights=weights,
        ) * np.log(2)

    # FIXME (from original R source) we are not allowing missing values, so df.residual must be same for all tags
    fit.df_residual = np.full(ntag, nlib - design.shape[1])
    fit.design = design
    fit.offset = offset
    fit.dispersion = dispersion
    fit.weights = weights
    fit.prior_count = prior_count
    return fit
