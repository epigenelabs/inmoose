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

# This file is based on the file 'R/aveLogCPM.R' of the Bioconductor edgeR
# package (version 3.38.4).
# This file contains a Python port of the original C++ code from the file
# 'src/R_ave_log_cpm.cpp' of the Bioconductor edgeR package (version 3.38.4).


from inspect import signature

import numpy as np

from .addPriorCount import add_prior_count
from .glm_one_group import glm_one_group
from .makeCompressedMatrix import (
    _compressDispersions,
    _compressOffsets,
    _compressPrior,
    _compressWeights,
)
from .mglmOneGroup import mglmOneGroup
from .utils import _isAllZero


def aveLogCPM_DGEList(self, normalized_lib_sizes=True, prior_count=2, dispersion=None):
    """
    Compute average log2 counts per million for each row of counts.

    See also
    --------
    This function calls :func:`aveLogCPM`.

    Arguments
    ---------
    self : DGEList
        DGEList object
    normalized_lib_sizes : bool, optional
        whether to use normalized library sizes. Defaults to :code:`True`.
    prior_count : float or array_like, optional
        scalar or vector of length :code:`self.counts.shape[0]`, containing the
        average value(s) to be added to each count to avoid infinite value on
        the log-scale. Defaults to :code:`2`.
    dispersion : float or array_like, optional
        scalar or vector of negative binomial dispersions.

    Returns
    -------
    ndarray
        numeric vector giving :code:`log2(AveCPM)` for each row of :code:`y`
    """
    # Library sizes should be stored in y but are sometimes missing
    lib_size = self.samples["lib_size"]
    if (lib_size.values == None).any():  # noqa: E711
        lib_size = self.counts.sum(axis=0)

    # Normalization factors should be stored in y but are sometimes missing
    if normalized_lib_sizes:
        nf = self.samples["norm_factors"]
        if (nf.values != None).all():  # noqa: E711
            lib_size = lib_size * nf

    # Dispersion supplied as argument takes precedence over value in object
    # Should trended_dispersion or tagwise_dispersion be used instead of common_dispersion if available?
    if dispersion is None:
        dispersion = self.common_dispersion

    return aveLogCPM(
        self.counts,
        lib_size=lib_size,
        prior_count=prior_count,
        dispersion=dispersion,
        weights=self.weights,
    )


def aveLogCPM_DGEGLM(y, prior_count=2, dispersion=None):
    """
    Compute average log2 counts per million for each row of counts.

    See also
    --------
    This function calls :func:`aveLogCPM`.

    Arguments
    ---------
    y : DGEGLM
        DGEGLM object
    prior_count : float or array_like, optional
        scalar or vector of length :code:`y.counts.shape[0]`, containing the
        average value(s) to be added to each count to avoid infinite value on
        the log-scale. Defaults to :code:`2`.
    dispersion : float or array_like, optional
        scalar or vector of negative binomial dispersions.

    Returns
    -------
    ndarray
        numeric vector giving :code:`log2(AveCPM)` for each row of :code:`y`
    """
    # Dispersion supplied as argument over-rules value in object
    if dispersion is None:
        dispersion = y.dispersion

    return aveLogCPM(
        y.counts,
        offset=y.offset,
        prior_count=prior_count,
        dispersion=dispersion,
        weights=y.weights,
    )


def aveLogCPM(
    y, lib_size=None, offset=None, prior_count=2, dispersion=None, weights=None
):
    """
    Compute average log2 counts per million for each row of counts.

    This function uses :func:`mglmOneGroup` to compute average counts per
    million (AveCPM) for each row of counts, and returns :code:`log2(AveCPM)`.
    An average value of :code:`prior_count` is added to the counts before
    running :func:`mglmOneGroup`. If :code:`prior_count` is a vector, each entry
    will be added to all counts in the corresponding row of :code:`y`, as
    described in :func:`addPriorCount`.

    This function is similar to :code:`log2(rowMeans(cpm(y, ...)))`, but with
    the refinement that larger library sizes are given more weight in the
    average. The two version will agree for large value of the dispersion.

    See also
    --------
    cpm : for individual logCPM values, rather than genewise averages
    addPriorCount : uses the same strategy to add the prior counts
    mglmOneGroup : computations for this function rely on :func:`mglmOneGroup`

    Arguments
    ---------
    y : matrix
        matrix of counts. Rows for genes and columns for libraries.
    lib_size : array_like, optional
        vector of library sizes. Defaults to :code:`np.sum(y, axis=0)`. Ignored
        if :code:`offset` is not :code:`None`.
    offset : matrix, optional
        matrix of offsets for the log-linear models. Defaults to :code:`None`.
    prior_count : float or array_like, optional
        scalar or vector of length :code:`y.shape[0]`, containing the average
        value(s) to be added to each count to avoid infinite value on the
        log-scale. Defaults to :code:`2`.
    dispersion : float or array_like, optional
        scalar or vector of negative binomial dispersions.
    weights : matrix, optional
        matrix of observation weights

    Returns
    -------
    ndarray
        numeric vector giving :code:`log2(AveCPM)` for each row of :code:`y`
    """
    y = np.asarray(y)
    if y.ndim != 2:
        raise ValueError("y should be a matrix")
    if y.shape[0] == 0:
        return 0

    # Special case when all counts and library sizes are zero
    if _isAllZero(y):
        if (lib_size is None or max(lib_size) == 0) and (
            offset is None or max(offset) == -np.inf
        ):
            abundance = np.full((y.shape[0],), -np.log(y.shape[0]))
            return (abundance + np.log(1e6)) / np.log(2)

    # Check dispersion
    if dispersion is None:
        dispersion = 0.05
    isna = np.isnan(dispersion)
    if isna.all():
        dispersion = 0.05
    elif isna.any():
        dispersion = np.asanyarray(dispersion)
        dispersion[isna] = np.nanmean(dispersion)

    dispersion = _compressDispersions(y, dispersion)

    # Check weights
    weights = _compressWeights(y, weights)

    # Check offsets
    offset = _compressOffsets(y, lib_size=lib_size, offset=offset)

    # Check prior counts
    prior_count = _compressPrior(y, prior_count)

    # Retrieve GLM fitting parameters
    maxit = signature(mglmOneGroup).parameters["maxit"].default
    tol = signature(mglmOneGroup).parameters["tol"].default

    # adding the current set of priors
    (prior_y, prior_offsets) = add_prior_count(y, offset, prior_count)
    # fitting a one-way layout
    fit = glm_one_group(
        prior_y,
        prior_offsets,
        dispersion,
        weights,
        maxit,
        tol,
        np.repeat(np.nan, y.shape[0]),
    )
    return (fit[0] + np.log(1e6)) / np.log(2)
