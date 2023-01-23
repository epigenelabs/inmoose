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


import logging
import numpy as np


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
    Estimate size factors.

    Arguments
    ---------
    obj : DESeqDataSet
        the input dataset
    type_ : "ratio", "poscounts" or "iterate"
        the algorithm to estimate the size factors
    locfunc
        defaults to the median
    geoMeans : array-like, optional
        user-provided geometric means
    controlGenes : ??
    normMatrix : ??
    quiet : bool
        controls verbosity, defaults to False
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
            logging.info(
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
            logging.info(
                "using 'normMatrix', adding normalization factors which correct for library size"
            )

    return obj


def estimateSizeFactorsForMatrix(
    counts, type_="ratio", locfunc=np.median, geoMeans=None, controlGenes=None
):
    """
    Arguments
    ---------
    counts : pandas.DataFrame
        raw counts. One column per gene, one row per sample.
    type_ : "ratio" or "poscounts"
        the algorithm to estimate the size factors
    locfunc
        a function to compute a location for a sample. By default, the median is used.
    geoMeans : ??
        gene-wise geometric mean (i.e. by column)
        optional
    controlGenes : ??
        optional
    """

    if type_ not in ["ratio", "poscounts"]:
        raise ValueError(f"type_ must be either 'ratio' or 'poscounts': {type_}")

    if geoMeans is None:
        incomingGeoMeans = False
        if type_ == "ratio":
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
        loggeomeans = np.log(geoMeans)

    if np.isinf(loggeomeans).all():
        raise ValueError(
            "every gene contains at least one zero, cannot compute log geometric means"
        )

    if controlGenes is None:

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
    """
    Arguments
    ---------
    counts : pandas.DataFrame
    """

    sf = estimateSizeFactorsForMatrix(
        counts / normMatrix,
        locfunc=locfunc,
        geoMeans=geoMeans,
        controlGenes=controlGenes,
    )
    nf = normMatrix * sf.reshape(sf.shape[0], 1)
    return nf / np.exp(np.mean(np.log(nf), 0))
