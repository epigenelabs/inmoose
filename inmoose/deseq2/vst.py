# -----------------------------------------------------------------------------
# Copyright (C) 2013-2022 Michael I. Love, Constantin Ahlmann-Eltze
# Copyright (C) 2023 Maximilien Colange

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

# This file is based on the file 'R/vst.R' Bioconductor DESeq2 package (version
# 3.16).


import numpy as np
from anndata import AnnData
from scipy.interpolate import CubicSpline

from . import (
    DESeqDataSet,
    DESeqTransform,
    estimateDispersionsFit,
    estimateDispersionsGeneEst,
)


def varianceStabilizingTransformation(obj, blind=True, fitType="parametric"):
    """
    Apply a variance stabilizing transformation (VST) to the count data

    This function calculates a variance stabilizing transformation (VST) from
    the fitted dispersion-mean relation(s) and then transforms the count data
    (normalized by division by the size factors or normalization factors),
    yielding a matrix of values which are now approximately homoskedastic
    (having constant variance along the range of mean values). The
    transformation also normalizes with respect to library size. The
    :func:`rlog` is less sensitive to size factors, which can be an issue when
    size factors vary widely. These transformations are useful when checking
    for outliers or as input for machine learning techniques such as clustering
    or linear discriminant analysis.

    Variance stabilizing transformation was originally described in [Anders2010]_.

    Details
    -------
    For each sample (*i.e.* line of :code:`dds.counts()`), the full variance
    function is calculated from the raw variance (by scaling according to the
    size factor and adding the shot noise). We recommend a blind estimation of
    the variance function, *i.e.* one ignoring conditions. This is performed by
    default, and can be modified using the :code:`blind` argument.

    Note that neither :func:`rlog` transformation nor the VST are used by the
    differential expression estimation in :func:`DESeq`, which always occurs on
    the raw count data, through generalized linear modeling which incorporates
    knowledge of the variance-mean dependence. The :func:`rlog` transformation
    and VST are offered as separate functionality which can be used for
    visualization, clustering or other machine learning tasks. See the
    transformation section of the vignette for more details.

    The transformation does not require that one has already estimated size
    factors and dispersions.

    A typical workflow is shown in Section *Variance stabilizing
    transformation* in the vignette.

    If :func:`estimateDispersions` was called with:

    - :code:`fitType="parametric"`: a closed-form expression for the variance
      stabilizing transformation is used on the normalized count data.
    - :code:`fitType="local"`: the reciprocal of the square root of the
      variance of the normalized counts, as derived from the dispersion fit, is
      then numerically integrated, and the integral (approximated by a spline
      function) is evaluated for each count value in the column, yielding a
      transformed value.
    - :code:`fitType="mean"`, a VST is applied for Negative Binomial
      distributed counts, :math:`k`, with a fixed dispersion, :math:`a`:
      :math:`(2 \\operatorname{asinh}(\sqrt{a k}) - \log(a) - \log(4)) / \log(2)`.

    In all cases, the transformation is scaled such that for large counts, it
    becomes asymptotically (for large values) equal to the logarithm to base 2
    of normalized counts.

    The variance stabilizing transformation from a previous dataset can be
    "frozen" and reapplied to new samples. The frozen VST is accomplished by
    saving the dispersion function accessible with :func:`dispersionFunction`,
    assigning this to the :class:`DESeqDataSet` with the new samples, and
    running :func:`varianceStabilizingTransformation` with :code:`blind` set to
    :code:`False`. Then the dispersion function from the previous dataset will
    be used to transform the new sample(s).

    Limitations: In order to preserve normalization, the same transformation
    has to be used for all samples. This results in the variance stabilization
    to be only approximate. The more the size factors differ, the more residual
    dependence of the variance on the mean will be found in the transformed
    data. :func:`rlog` is a transformation which can perform better in these
    cases. As shown in the vignette, :code:`meanSdPlot` from the package *vsn*
    can be used to see whether this is a problem.

    Arguments
    ---------
    obj : DESeqDataSet or matrix
        a :class:`DESeqDataSet` or matrix of counts
    blind : bool
        whether to blind the transformation to the experimental design.
        :code:`blind=True` should be used for comparing samples in a manner
        unbiased by prior information on samples, for example to perform sample
        QA (quality assurance).
        :code:`blind=False` should be used for transforming data for downstream
        analysis, where the full use of the design information should be made.
        :code:`blind=False` will skip re-estimation of the dispersion trend, if
        this has already been calculated.
        If many genes have large differences in counts due to the experimental design,
        it is important to set :code:`blind=False` for downstream analysis.
        Defaults to :code:`True`.
    fitType : { "parametric", "local", "mean" }
        in case dispersions have not yet been estimated for :code:`self`, this
        parameter is passed on to :func:`estimateDispersions` (options described
        there).
        Defaults to :code:`"parametric"`.

    Returns
    -------
    DESeqTransform or matrix
        returns a :class:`DESeqTransform` if a :class:`DESeqDataSet` was
        provided, or returns a matrix if a count matrix was provided.
        Note that for :class:`DESeqTransform` output, the matrix of transformed
        values is stored in :code:`vsd.layers`.
    """
    if not isinstance(obj, DESeqDataSet):
        matrixIn = True
        obj = DESeqDataSet(obj, design="~1")
    else:
        obj = obj.copy()
        matrixIn = False

    if obj.sizeFactors is None and obj.normalizationFactors is None:
        obj = obj.estimateSizeFactors()

    if blind:
        obj.design = "~1"

    if blind or obj.dispersionFunction.fitType is None:
        obj = estimateDispersionsGeneEst(obj, quiet=True)
        obj = estimateDispersionsFit(obj, fitType, quiet=True)

    vsd = getVarianceStabilizedData(obj)
    if matrixIn:
        return vsd

    # TODO missing metadata
    ad = AnnData(vsd, obs=obj.obs, var=obj.var)
    return DESeqTransform(ad)


def getVarianceStabilizedData(obj):
    if obj.dispersionFunction.fitType is None:
        raise ValueError(
            "call estimateDispersions before calling getVarianceStabilizedData"
        )

    ncounts = obj.counts(normalized=True)
    if obj.dispersionFunction.fitType == "parametric":
        coefs = obj.dispersionFunction.coefficients

        def vst_fn(q):
            return np.log(
                (
                    1
                    + coefs["extraPois"]
                    + 2 * coefs["asymptDisp"] * q
                    + 2
                    * np.sqrt(
                        coefs["asymptDisp"]
                        * q
                        * (1 + coefs["extraPois"] + coefs["asymptDisp"] * q)
                    )
                )
                / (4 * coefs["asymptDisp"])
            ) / np.log(2)

        return vst_fn(ncounts)

    elif obj.dispersionFunction.fitType == "local":
        # non-parametric fit -> numerical integration
        if obj.sizeFactors is None:
            if obj.normalizationFactors is None:
                raise ValueError(
                    "both sizeFactors and normalizationFactors are missing!"
                )
            sf = np.exp(np.log(obj.normalizationFactors).mean(axis=1))
        else:
            sf = obj.sizeFactors
        xg = np.sinh(np.linspace(np.arcsinh(0), np.arcsinh(ncounts.max()), num=1000))[
            1:
        ]
        xim = (1 / sf).mean()
        baseVarsAtGrid = obj.dispersionFunction(xg) * xg**2 + xim * xg
        integrand = 1 / np.sqrt(baseVarsAtGrid)
        splf = CubicSpline(
            np.arcsinh((xg[1:] + xg[:-1]) / 2),
            ((xg[1:] - xg[:-1]) * (integrand[1:] + integrand[:-1]) / 2).cumsum(),
        )
        h1 = np.quantile(ncounts.mean(axis=1), 0.95)
        h2 = np.quantile(ncounts.mean(axis=1), 0.999)
        eta = (np.log2(h2) - np.log2(h1)) / (
            splf(np.arcsinh(h2)) - splf(np.arcsinh(h1))
        )
        xi = np.log2(h1) - eta * splf(np.arcsinh(h1))
        tc = [
            eta * splf(np.arcsinh(ncounts[:, clm])) + xi
            for clm in obj.counts().var_names
        ]
        tc.rownames = obj.counts().rownames
        return tc
    elif obj.dispersionFunction.fitType == "mean":
        alpha = obj.dispersionFunction.mean
        # the following stabilizes NB counts with fixed dispersion alpha
        # and converges to log2(q) as q => infinity

        def vst_fn(q):
            return (
                2 * np.arcsinh(np.sqrt(alpha * q)) - np.log(alpha) - np.log(4)
            ) / np.log(2)

        return vst_fn(ncounts)
    else:
        raise ValueError("fitType is not parametric, local or mean")


def vst(obj, blind=True, nsub=1000, fitType="parametric"):
    """
    Quickly estimate dispersion trend and apply a variance stabilizing transformation

    This is a wrapper for the :func:`varianceStabilizingTransformation` (VST)
    that provides much faster estimation of the dispersion trend used to
    determine the formula for the VST. The speed-up is accomplished by
    subsetting to a smaller number of genes in order to estimate this
    dispersion trend. The subset of genes is chosen deterministically, to span
    the range of genes' mean normalized counts.

    Arguments
    ---------
    obj : DESeqDataSet or matrix
        a :class:`DESeqDataSet` or matrix of counts
    blind : bool
        whether to blind the transformation to the experimental design (see
        :func:`varianceStabilizingTransformation`)
    nsub : int
        the number of genes to subset to (default 1000)
    fitType : { "parametric", "local", "mean" }
        for estimation of dispersions: this parameter is passed on to
        :func:`estimateDispersions` (options described there)

    Returns
    -------
    DESeqTransform or matrix
        same as :func:`varianceStabilizingTransformation`
    """
    if obj.n_var < nsub:
        raise ValueError(
            f"Object has less than {nsub} rows, it is recommended to use varianceStabilizingTransformation directly"
        )

    if not isinstance(obj, DESeqDataSet):
        matrixIn = True
        obj = DESeqDataSet(obj, design="~1")
    else:
        if blind:
            obj.design = "~1"
        matrixIn = False

    if obj.sizeFactors is None and obj.normalizationFactors is None:
        obj = obj.estimateSizeFactors()
    baseMean = obj.counts(normalized=True).mean(axis=0)
    if (baseMean > 5).sum() < nsub:
        raise ValueError(
            "Object has less than {nsub} genes with mean normalized count > 5, it is recommended to use varianceStabilizingTransformation directly."
        )

    # subset to a specified number of genes with mean normalized count > 5
    obj_sub = obj[:, baseMean > 5]
    baseMean = baseMean[baseMean > 5]
    o = np.argsort(baseMean)
    idx = o[np.linspace(0, len(o) - 1, num=nsub, dtype=int)]
    obj_sub = obj_sub[idx, :]

    # estimate dispersion trend
    obj_sub = estimateDispersionsGeneEst(obj_sub, quiet=True)
    obj_sub = estimateDispersionsFit(obj_sub, fitType=fitType, quiet=True)

    # assign to the full object
    obj.setDispFunction(obj_sub.dispersionFunction)

    # calculate and apply the VST (note blinding is accomplished above, here
    # :code:`blind=False` is used to avoid re-calculating dispersion)
    vsd = varianceStabilizingTransformation(obj, blind=False)
    if matrixIn:
        return vsd.X
    else:
        return vsd
