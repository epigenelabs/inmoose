#-----------------------------------------------------------------------------
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
#-----------------------------------------------------------------------------

# This file is based on the file 'R/dispCoxReidInterpolateTagwise.R' of the Bioconductor edgeR package (version 3.38.4).


import numpy as np
from math import floor

from .adjustedProfileLik import adjustedProfileLik
from .aveLogCPM import aveLogCPM
from .makeCompressedMatrix import _compressOffsets, _compressWeights
from .maximizeInterpolant import maximizeInterpolant
from .movingAverageByCol import movingAverageByCol
from edgepy_cpp import cxx_maximize_interpolant

def dispCoxReidInterpolateTagwise(y, design, dispersion, offset=None, trend=True, AveLogCPM=None, min_row_sum=5, prior_df=10, span=0.3, grid_npts=11, grid_range=(-6,6), weights=None):
    """
    Estimate tagwise NB dispersions using weighted Cox-Reid Adjusted Profile-likelihood and cubic spline interpolation over a tagwise grid.
    """
    # Check y
    y = np.asarray(y, order='F')
    (ntags, nlibs) = y.shape

    # Check design
    design = np.asarray(design, order='F')
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
    if len(offset.shape) == 1:
        offset = np.full(y.shape, offset, order='F')
    assert offset.shape == y.shape

    # Check AveLogCPM
    if AveLogCPM is None:
        AveLogCPM = aveLogCPM(y, lib_size=lib_size)
    AveLogCPM = np.asarray(AveLogCPM, order='F')

    # Check dispersion
    dispersion = np.asanyarray(dispersion, order='F')
    if len(dispersion.shape) == 0:
        dispersion = np.full((ntags,), dispersion, order='F')
    else:
        if len(dispersion) != ntags:
            raise ValueError("length of dispersion does not match nrow(y)")

    # Apply min_row_sum and use input dispersion for small count tags
    i = y.sum(axis=1) >= min_row_sum
    if np.logical_not(i).any():
        if i.any():
            dispersion[i] = dispCoxReidInterpolateTagwise(y=y[i,:], design=design, offset=offset[i,:], dispersion=dispersion[i], AveLogCPM=AveLogCPM[i], grid_npts=grid_npts, min_row_sum=0, prior_df=prior_df, span=span, trend=trend, weights=(weights[i,:] if weights is not None else None))
        return dispersion

    # Posterior profile likelihood
    prior_n = prior_df / (nlibs-ncoefs)
    spline_pts = [grid_range[0] + i*(grid_range[1]-grid_range[0])/(grid_npts-1) for i in range(grid_npts)]
    apl = np.zeros((ntags, grid_npts), order='F')
    # anticipate the calls to _compress* in adjustedProfileLik
    y = np.asarray(y, order='F')
    offset = _compressOffsets(y, offset=offset)
    weights = _compressWeights(y, weights)
    for i in range(grid_npts):
        spline_disp = dispersion * 2**spline_pts[i]
        apl[:,i] = adjustedProfileLik(spline_disp, y=y, design=design, offset=offset, weights=weights)

    if trend:
        o = np.argsort(AveLogCPM)
        oo = np.argsort(o)
        width = floor(span * ntags)
        apl_smooth = movingAverageByCol(apl[o,:], width=width)[oo,:]
    else:
        apl_smooth = np.full((ntags, grid_npts), apl.mean(axis=0), order='F')
    apl_smooth = (apl + prior_n*apl_smooth) / (1+prior_n)

    # Tagwise maximization
    d = maximizeInterpolant(spline_pts, apl_smooth)
    d = np.asarray(d, order='F')
    return dispersion * 2**d