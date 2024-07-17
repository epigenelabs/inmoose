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

# This file is based on the file 'R/core.R' of the Bioconductor DESeq2 package
# (version 3.16).


import numpy as np

from ..utils import LOGGER


def estimateSizeFactors_dds(
    obj,
    type_="ratio",
    locfunc=np.median,
    geoMeans=None,
    controlGenes=None,
    normMatrix=None,
    quiet=False,
):
    """
    Estimate the size factors of a :class:`DESeqDataSet`

    This function estimates the size factors using the "median ratio method",
    described by Equation 5 in [Anders2010]_.

    The estimated size factors can be accessed through the
    :attr:`DESeqDataSet.sizeFactors` property of :class:`DESeqDataSet`.
    Alternative library size estimators can also be supplied through this
    property.

    See :func:`.DESeq` for a description of the use of size factors in the GLM.
    One should call this function after building a :obj:`DESeqDataSet`, unless
    size factors are manually specified with property
    :attr:`DESeqDataSet.sizeFactors`.  Alternatively, gene-specific
    normalization factors for each sample can be provided using the
    :attr:`DESeqDataSet.normalizationFactors` which will always preempt
    :attr:`DESeqDataSet.sizeFactors` in calculations.

    Internally, the function calls :func:`.estimateSizeFactorsForMatrix`, which
    provides more details on the calculation.

    See also
    --------
    .DESeq
    .estimateSizeFactorsForMatrix

    Arguments
    ---------
    obj : DESeqDataSet
        the input dataset
    type_ : "ratio", "poscounts" or "iterate"
        the algorithm to estimate the size factors

        :code:`"ratio"` uses the standard median ratio method introduced in
        DESeq. The size factor is the median ratio of the sample over a
        "pseudosample": for each gene, the geometric mean of all samples.

        :code:`"poscounts"` and :code:`"iterate"` offer alternative estimators,
        which can be used even when all genes contain a sample with a zero (a
        problem for the default method, as the geometric then becomes zero, and
        the ratio undefined).

        The :code:`"poscounts"` estimator deals with a gene with some zeros by
        calculating a modified geometric mean by taking the n-th root of the
        product of the non-zero counts.  This evolved out of use cases with
        Paul McMurdie's phyloseq package for metagenomic samples.

        The :code:`"iterate"` estimator iterates between estimating the
        dispersion with a design of ~1, and finding a size factor vector by
        numerically optimizing the likelihood of the ~1 model.
    locfunc
        a function to compute a location for a sample. By default, the median is
        used.
    geoMeans : array-like, optional
        by default, the geometric means of the counts are calculated within the
        function. A vector of geometric means from another count matrix can be
        provided for a "frozen" size factor calculation. The size factors will
        be scaled to have a geometric mean of 1 when supplying :code:`geoMeans`.
    controlGenes : array-like, optional
        index vector specifying those genes to use for size factor estimation
        (e.g. housekeeping or spike-in genes)
    normMatrix : ndarray, optional
        a matrix of normalization factors which do not yet control for library
        size. Providing :code:`normMatrix` will estimate size factors on the
        count matrix divided by :code:`normMatrix` and store the product of the
        size factors and :code:`normMatrix` as
        :attr:`DESeqDataSet.normalizationFactors`.  It is recommended to divide
        out the sample-wise geometric mean of :code:`normMatrix` so the
        sample-wise factors are roughly centered on 1.
    quiet : bool
        controls verbosity, defaults to False

    Returns
    -------
    DESeqDataSet
        the input :code:`obj`, with the size factors filled in
    """

    if type_ not in ["ratio", "poscounts", "iterate"]:
        raise ValueError(f"invalid type: {type_}")

    if type_ == "iterate":
        raise NotImplementedError(
            "iterative method to estimate size factors not implemented yet"
        )
        obj.sizeFactors = obj.estimateSizeFactorsIterate()
    else:
        if type_ == "poscounts":

            def geoMeanNZ(x):
                if (x == 0).all():
                    return 0
                else:
                    return np.exp(np.sum(np.log(x[x > 0])) / len(x))

            geoMeans = np.apply_along_axis(geoMeanNZ, 0, obj.X)

        if "avgTxLength" in obj.obsm.keys():
            nm = obj.obsm["avgTxLength"]
            nm = nm / np.exp(np.mean(np.log(nm), 1))
            obj.normalizationFactors = estimateNormFactors(
                obj.X,
                normMatrix=nm,
                locfunc=locfunc,
                geoMeans=geoMeans,
                controlGenes=controlGenes,
            )
            LOGGER.info(
                "using 'avgTxLength' from dds.obsm, correcting for library size"
            )

        elif normMatrix is None:
            obj.sizeFactors = estimateSizeFactorsForMatrix(
                obj.X, locfunc=locfunc, geoMeans=geoMeans, controlGenes=controlGenes
            )
        else:
            obj.normalizationFactors = estimateNormFactors(
                obj.X,
                normMatrix=normMatrix,
                locfunc=locfunc,
                geoMeans=geoMeans,
                controlGenes=controlGenes,
            )
            LOGGER.info(
                "using 'normMatrix', adding normalization factors which correct for library size"
            )

    return obj


def estimateSizeFactorsForMatrix(
    counts, type_="ratio", locfunc=np.median, geoMeans=None, controlGenes=None
):
    """
    Low-level function to estimate size factors with robust regression

    Given a matrix or data frame of count data, this function estimates the
    size factors as follows: each row is divided by the geometric means of the
    columns. The median (or, if requested, another location estimator) of these
    ratios (skipping the genes with a geometric mean of zero) is used as the
    size factor for this row. Typically, one will not call this function
    directly, but use :meth:`.DESeqDataSet.estimateSizeFactors`.

    See also
    --------
    .DESeqDataSet.estimateSizeFactors

    Arguments
    ---------
    counts : array-like
        matrix of raw counts. One column per gene, one row per sample.
    type_ : "ratio" or "poscounts"
        the algorithm to estimate the size factors: standard median ratio
        (:code:`"ratio"`), or there the geometric mean is only calculated over
        positive counts (:code:`"poscounts"`).
    locfunc
        a function to compute a location for a sample. By default, the median
        is used.
    geoMeans : ndarray, optional
        by default, the geometric means of the counts are calculated within
        the function. A vector of geometric means from another count matrix can
        be provided for a "frozen" size factor calculation.
    controlGenes : array-like, optional
        index vector specifying those genes to use for size factor estimation
        (e.g. housekeeping or spike-in genes)

    Returns
    -------
    ndarray
        the estimated size factors, one element per row of :code:`counts`
    """

    if type_ not in ["ratio", "poscounts"]:
        raise ValueError(f"type_ must be either 'ratio' or 'poscounts': {type_}")

    if geoMeans is None:
        incomingGeoMeans = False
        if type_ == "ratio":
            with np.errstate(divide="ignore"):
                loggeomeans = np.mean(np.log(counts), 0)
        elif type_ == "poscounts":
            lc = np.log(counts)
            lc[~np.isfinite(lc)] = 0
            loggeomeans = np.mean(lc, 0)
            allZero = np.sum(counts, 0) == 0
            loggeomeans[allZero] = -np.inf

    else:
        incomingGeoMeans = True
        if len(geoMeans) != counts.shape[1]:
            raise ValueError(
                "geoMeans should be as long as the number of columns of counts"
            )
        with np.errstate(divide="ignore"):
            loggeomeans = np.log(geoMeans)

    if np.isinf(loggeomeans).all():
        raise ValueError(
            "every gene contains at least one zero, cannot compute log geometric means"
        )

    if controlGenes is None:

        @np.errstate(invalid="ignore", divide="ignore")
        def sf_compute(cnts):
            return np.exp(
                locfunc(
                    (np.log(cnts) - loggeomeans)[np.isfinite(loggeomeans) & (cnts > 0)]
                )
            )

        sf = np.apply_along_axis(sf_compute, 1, counts)
    else:
        loggeomeansSub = loggeomeans[controlGenes]

        def sf_compute(cnts):
            return np.exp(
                locfunc(
                    (np.log(cnts) - loggeomeansSub)[
                        np.isfinite(loggeomeansSub) & (cnts > 0)
                    ]
                )
            )

        sf = np.apply_along_axis(sf_compute, 1, counts[:, controlGenes])

    if incomingGeoMeans:
        # stabilize the size factors to have geometric mean of 1
        sf = sf / np.exp(np.mean(np.log(sf)))

    return sf


def estimateNormFactors(
    counts, normMatrix, locfunc=np.median, geoMeans=None, controlGenes=None
):
    sf = estimateSizeFactorsForMatrix(
        counts / normMatrix,
        locfunc=locfunc,
        geoMeans=geoMeans,
        controlGenes=controlGenes,
    )
    nf = normMatrix * sf.reshape(sf.shape[0], 1)
    return nf / np.exp(np.mean(np.log(nf), 0))
