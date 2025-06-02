# -----------------------------------------------------------------------------
# Copyright (C) 2008-2022 Yunshun Chen, Aaron TL Lun, Davis J McCarthy, Matthew E Ritchie, Belinda Phipson, Yifang Hu, Xiaobei Zhou, Mark D Robinson, Gordon K Smyth
# Copyright (C) 2022-2025 Maximilien Colange

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
import pandas as pd
import scipy
from patsy import DesignMatrix, dmatrix

from ..utils import asfactor
from .DGEGLM import DGEGLM
from .DGELRT import DGELRT
from .makeCompressedMatrix import _compressDispersions, _compressOffsets
from .mglmLevenberg import mglmLevenberg
from .mglmOneWay import designAsFactor, mglmOneWay
from .nbinomDeviance import nbinomDeviance
from .predFC import predFC


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
    DGEGLM
        object containing the data about the fit
    """

    # The design matrix defaults to the oneway layout defined by self.samples["group"]
    # If there is only one group, then the design matrix is left None so that a matrix with a single intercept column will be set later by glmFit.
    if design is None:
        design = self.design
        if design is None:
            group = asfactor(self.samples["group"]).droplevels()
            if group.nlevels() > 1:
                design = dmatrix("~C(self.samples['group'])")

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

    This function implements one of the GLM methods developed by [McCarthy2012]_.

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
    y : pd.DataFrame
        matrix of counts
    design : matrix, optional
        design matrix for the genewise linear models. Must be of full column
        rank. Defaults to a single column of ones, equivalent to treating the
        columns as replicate libraries.
    dispersion : float or array_like
        scalar, vector or matrix of negative binomial dispersions. Can be a
        common value for all genes, a vector of dispersion values with one for
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
    """
    # Check y
    (ntag, nlib) = y.shape

    # Check design
    if design is None:
        design = dmatrix("~1", pd.DataFrame(y.T))
    try:
        design = DesignMatrix(
            np.asarray(design, order="F"), design_info=design.design_info
        )
    except AttributeError:
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
    dispersion = np.asanyarray(dispersion)
    # TODO check dispersion for NaN and non-numeric values
    if dispersion.shape not in [(), (1,), (ntag,), y.shape]:
        raise ValueError("Dimensions of dispersion do not agree with dimensions of y")
    dispersion_mat = _compressDispersions(y, dispersion)

    # Check offset
    if offset is not None:
        # TODO check that offset is numeric
        offset = np.asanyarray(offset)
        if offset.shape not in [(), (1,), (nlib,), y.shape]:
            raise ValueError("Dimensions of offset do not agree with dimensions of y")

    # Check lib_size
    if lib_size is not None:
        # TODO check that lib_size is numeric
        lib_size = np.asarray(lib_size)
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

    # counts N,M
    # design M,P
    assert y.shape[1] == design.shape[0]
    w_vec = fit.fitted_values / (1.0 + dispersion_mat * fit.fitted_values)
    if weights is not None:
        w_vec = weights * w_vec
    ridge = np.diag(np.repeat(1e-6 / (np.log(2) ** 2), design.shape[1]))
    xtwxr_inv = np.linalg.inv(design.T @ (design * w_vec[:, :, None]) + ridge)
    sigma = xtwxr_inv @ design.T @ (design * w_vec[:, :, None]) @ xtwxr_inv
    fit.coeff_SE = np.diagonal(sigma, axis1=-2, axis2=-1)

    # FIXME (from original R source) we are not allowing missing values, so df.residual must be same for all tags
    fit.df_residual = np.full(ntag, nlib - design.shape[1])
    fit.design = design
    fit.offset = offset
    fit.dispersion = dispersion
    fit.weights = weights
    fit.prior_count = prior_count
    return fit


def glmLRT(glmfit, coef=None, contrast=None):
    """
    Conduct genewise statistical tests for a given coefficient or coefficient contrast.

    This function implements one of the GLM methods developed by [McCarthy2012]_.

    :func:`glmLRT` conducts likelihood ratio tests for one or more coefficients
    in the linear model. If :code:`coef` is used, the null hypothesis is that
    all the coefficients indicated by :code:`coef` are equal to zero. If
    :code:`contrast` is non-null, then the null hypothesis is that the
    specified contrasts of the coefficients are equal to zero. For example, a
    contrast of :code:`[0,1,-1]`, assuming there are three coefficients, would
    test the hypothesis that the second and third coefficients are equal.

    Arguments
    ---------
    glmfit : DGEGLM
        a :class:`DGEGLM` object, usually output from :func:`glmFit`
    coef : array_like of integers or strings
        vector indicating which coefficients of the linear model are to be
        tested equal to zero. Values must be column indices or column names of
        :code:`design`. Defaults to the last coefficient. Ignored if
        :code:`contrast` is specified.
    contrast : array or matrix of integers
        vector or matrix specifying one or more contrasts of the linear model
        coefficients to be tested equal to zero. Number of rows must equal to
        the number of columns of :code:`design`. If specified, then takes
        precedence over :code:`coef`.

    Returns
    -------
    DGELRT
        dataframe with two additional components:

        - :code:`fit` containing the result of :func:`glmFit`
        - :code:`comparison`, string describing the coefficient or the contrast
          being tested

        The dataframe has the same rows as :code:`y` and is ready to be
        displayed by :func:`topTags`. It contains the following columns:

        - :code:`"log2FoldChange"`, log2-fold-change of expression between
          conditions being tested.
        - :code:`"lfcSE"`, standard error of log2-fold-change.
        - :code:`"logCPM"`, average log2-counts per million, the average taken
          over all libraries in :code:`y`.
        - :code:`"stat"`, likelihood ratio statistics.
        - :code:`"pvalue"`, *p*-values.
    """
    if coef is None:
        coef = glmfit.design.shape[1] - 1
    if not isinstance(glmfit, DGEGLM):
        raise ValueError("glmfit must be a DGEGLM object (usually produced by glmFit).")

    if glmfit.AveLogCPM is None:
        glmfit.AveLogCPM = glmfit.aveLogCPM()
    nlibs = glmfit.coefficients.shape[1]

    # check design matrix
    design = glmfit.design
    nbeta = design.shape[1]
    if nbeta < 2:
        raise ValueError(
            "Need at least two columns for design, usually the first is the intercept columns"
        )
    coef_names = np.array(design.design_info.column_names)

    # Evaluate logFC for coef to be tested
    # Note that contrast takes precedence over coef: if contrast is given then reform
    # design matrix so that contrast of interest is last column
    if contrast is None:
        if not isinstance(coef, (list, np.ndarray)):
            coef = [coef]
        if isinstance(coef[0], str):
            check_coef = np.isin(coef, design.design_info.column_names)
            if (~check_coef).any():
                raise ValueError(
                    "One or more named coef arguments do not match a column of the design matrix."
                )
            coef_name = coef
            coef = np.nonzero([design.design_info.column_names == c for c in coef])[0]
        else:
            coef_name = [coef_names[c] for c in coef]
        logFC = glmfit.coefficients[:, coef] / np.log(2)
        lfcSE = glmfit.coeff_SE[:, coef] / np.log(2)
    else:
        contrast = np.array(contrast)
        if contrast.ndim > 2:
            raise ValueError("contrast must be 1-D or 2-D")
        if contrast.ndim < 2:
            contrast = contrast.reshape(contrast.shape[0], 1)
        if contrast.shape[0] != glmfit.coefficients.shape[1]:
            raise ValueError(
                "contrast vector of wrong length, should be equal to number of coefficients in the linear model"
            )
        ncontrasts = np.linalg.matrix_rank(contrast)
        Q, R = np.linalg.qr(contrast, mode="complete")
        if ncontrasts == 0:
            raise ValueError("contrasts are all zero")
        coef = np.arange(ncontrasts)
        logFC = (glmfit.coefficients @ contrast) / np.log(2)
        lfcSE = (glmfit.coeff_SE @ contrast) / np.log(2)
        if ncontrasts > 1:
            coef_name = f"LR test on {ncontrasts} degrees of freedom"
        else:
            contrast = np.squeeze(contrast)
            i = contrast != 0
            coef_name = " ".join(
                [f"{a}*{b}" for a, b in zip(contrast[i], coef_names[i])]
            )
        Dvec = np.ones(nlibs, int)
        Dvec[coef] = np.diag(R)[coef]
        Q = Q * Dvec
        design = design @ Q

    # Null design matrix
    non_coef = np.setdiff1d(np.arange(design.shape[1]), coef)
    design0 = design[:, non_coef]

    # Null fit
    fit_null = glmFit(
        glmfit.counts,
        design=design0,
        offset=glmfit.offset,
        weights=glmfit.weights,
        dispersion=glmfit.dispersion,
        prior_count=0,
    )

    # Likelihood ratio statistic
    LR = np.subtract(fit_null.deviance, glmfit.deviance)
    df_test = fit_null.df_residual - glmfit.df_residual
    LRT_pvalue = scipy.stats.chi2.sf(LR, df=df_test)
    tab = pd.DataFrame()
    if logFC.ndim > 1:
        for i in range(logFC.shape[1]):
            tab[f"logFC{i}"] = logFC[:, i]
            tab[f"lfcSE{i}"] = lfcSE[:, i]
        tab.columns = [
            "log2FoldChange" if i % 2 == 0 else "lfcSE"
            for i in range(2 * logFC.shape[1])
        ]

    else:
        tab["log2FoldChange"] = logFC
        tab["lfcSE"] = lfcSE
    tab["logCPM"] = glmfit.AveLogCPM
    tab["stat"] = LR
    tab["pvalue"] = LRT_pvalue
    tab.index = glmfit.counts.index
    res = DGELRT(tab, glmfit)
    res.comparison = coef_name
    res.df_test = df_test
    return res
