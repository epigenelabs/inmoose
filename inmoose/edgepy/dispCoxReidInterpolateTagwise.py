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

# This file is based on the file 'R/dispCoxReidInterpolateTagwise.R' of the Bioconductor edgeR package (version 3.38.4).


from math import floor

import numpy as np

from .adjustedProfileLik import adjustedProfileLik
from .aveLogCPM import aveLogCPM
from .makeCompressedMatrix import _compressOffsets, _compressWeights
from .maximizeInterpolant import maximizeInterpolant
from .movingAverageByCol import movingAverageByCol


def dispCoxReidInterpolateTagwise(
    y,
    design,
    dispersion,
    offset=None,
    trend=True,
    AveLogCPM=None,
    min_row_sum=5,
    prior_df=10,
    span=0.3,
    grid_npts=11,
    grid_range=(-6, 6),
    weights=None,
):
    """
    Estimate genewise dispersion parameters across multiple negative binomial
    GLMs using weighted Cox-Reid adjusted profile likelihood and cubic spline
    interpolation over a genewise grid.

    In the context of :code:`edgepy`, :func:`dispCoxReidInterpolateTagwise` is a
    low-level function called by :func:`estimateGLMTagwiseDisp`.

    This function calls :func:`maximizeInterpolant` to fit cubic spline
    interpolation over a genewise grid.

    Note that the terms "tag" and "gene" are synonymous here. The function is
    only named "tagwise" for historical reasons.

    Arguments
    ---------
    y : matrix
        matrix of counts
    design : matrix
        design matrix for the GLM to fit
    dispersion : float or array_like
        scalar or vector giving the dispersion(s) towards which the genewise
        dispersion parametes are shrunk
    offset : float or array_like, optional
        scalar, vector or matrix giving the offset (in addition to the log of
        the effective library size) that is to be included in the NB GLM for the
        genes. If a scalar, then this value is used as an offset for all genes
        and libraries. If a vector, it should have length equal to the number of
        libraries, and the same vector of offsets is used for each gene. If a
        matrix, then each library for each gene has its unique offset. In
        :func:`adjustedProfileLik` the :code:`offset` must be a matrix with the
        same shape as the matrix of counts.
    trend : bool, optional
        whether abundance-dispersion trend is used for smoothing
    AveLogCPM : array_like, optional
        vector of average log2 counts per million for each gene
    min_row_sum : int, optional
        value to filter out low abundance genes. Only genes with total sum of
        counts above this threshold are used. Low abundance genes can adversely
        affect the estimation of the common dispersion, so this argument allows
        the user to select an appropriate filter threshold for gene abundance.
        Defaults to 5.
    prior_df : float, optional
        prior desmoothing parameter that indicates the weight to give to the
        common likelihood compared to the individual gene's likelihood; default
        :code:`getPriorN(obj)` gives a value for :code:`prior_n` that is
        equivalent to giving the common likelihood 20 prior degrees of freedom
        in the estimation of the genewise dispersion.
    span : float, optional
        parameter between 0 and 1 specifying proportion of data to be used in
        the local regression moving window. Larger values give smoother fits.
    grid_npts : int, optional
        the number of points at which to place knots for the spline-based
        estimation of the genewise dispersion estimates.
    grid_range : tuple, optional
        relative range, in terms of :code:`log2(dispersion)`, on either side of
        trendline for each gene for spline grid points.
    weights : matrix, optional
        observation weights

    Returns
    -------
    ndarray
        vector of genewise dispersion, same length as the number of genes in
        the input matrix of counts
    """
    # Check y
    y = np.asarray(y)
    (ntags, nlibs) = y.shape

    # Check design
    design = np.asarray(design)
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("design matrix must be full column rank")
    ncoefs = design.shape[1]
    if ncoefs >= nlibs:
        raise ValueError("no residual degrees of freedom")

    # Check offset
    lib_size = None
    if offset is None:
        lib_size = y.sum(axis=0)
        offset = np.log(lib_size)
    if offset.ndim == 1:
        offset = np.broadcast_to(offset, y.shape)
    assert offset.shape == y.shape

    # Check AveLogCPM
    if AveLogCPM is None:
        AveLogCPM = aveLogCPM(y, lib_size=lib_size)
    AveLogCPM = np.asarray(AveLogCPM)

    # Check dispersion
    dispersion = np.asanyarray(dispersion)
    if dispersion.ndim == 0:
        dispersion = np.broadcast_to(dispersion, (ntags,))
    else:
        if len(dispersion) != ntags:
            raise ValueError("length of dispersion does not match nrow(y)")

    # Apply min_row_sum and use input dispersion for small count tags
    i = y.sum(axis=1) >= min_row_sum
    if np.logical_not(i).any():
        if i.any():
            if not dispersion.flags.writeable:
                dispersion = dispersion.copy()
            dispersion[i] = dispCoxReidInterpolateTagwise(
                y=y[i, :],
                design=design,
                offset=offset[i, :],
                dispersion=dispersion[i],
                AveLogCPM=AveLogCPM[i],
                grid_npts=grid_npts,
                min_row_sum=0,
                prior_df=prior_df,
                span=span,
                trend=trend,
                weights=(weights[i, :] if weights is not None else None),
            )
        return dispersion

    # Posterior profile likelihood
    prior_n = prior_df / (nlibs - ncoefs)
    spline_pts = [
        grid_range[0] + i * (grid_range[1] - grid_range[0]) / (grid_npts - 1)
        for i in range(grid_npts)
    ]
    apl = np.zeros((ntags, grid_npts))
    # anticipate the calls to _compress* in adjustedProfileLik
    y = np.asarray(y)
    offset = _compressOffsets(y, offset=offset)
    weights = _compressWeights(y, weights)
    for i in range(grid_npts):
        spline_disp = dispersion * 2 ** spline_pts[i]
        apl[:, i] = adjustedProfileLik(
            spline_disp, y=y, design=design, offset=offset, weights=weights
        )

    if trend:
        o = np.argsort(AveLogCPM)
        oo = np.argsort(o)
        width = floor(span * ntags)
        apl_smooth = movingAverageByCol(apl[o, :], width=width)[oo, :]
    else:
        apl_smooth = np.full((ntags, grid_npts), apl.mean(axis=0))
    apl_smooth = (apl + prior_n * apl_smooth) / (1 + prior_n)

    # Tagwise maximization
    d = maximizeInterpolant(spline_pts, apl_smooth)
    d = np.asarray(d)
    return dispersion * 2**d
