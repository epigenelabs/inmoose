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

# This file is based on the file 'R/aveLogCPM.R' of the Bioconductor edgeR package (version 3.38.4).


from inspect import signature
import numpy as np

from .utils import _isAllZero
from .makeCompressedMatrix import _compressDispersions, _compressWeights, _compressOffsets, _compressPrior
from .mglmOneGroup import mglmOneGroup
from edgepy_cpp import cxx_ave_log_cpm

def aveLogCPM_DGEList(self, normalized_lib_sizes=True, prior_count=2, dispersion=None):
    """
    log2(AveCOM)
    """
    # Library sizes should be stored in y but are sometimes missing
    lib_size = self.samples.lib_size
    if (lib_size.values == None).any():
        lib_size = self.counts.sum(axis=0)

    # Normalization factors should be stored in y but are sometimes missing
    if normalized_lib_sizes:
        nf = self.samples.norm_factors
        if (nf.values != None).all():
            lib_size = lib_size*nf

    # Dispersion supplied as argument takes precedence over value in object
    # Should trended_dispersion or tagwise_dispersion be used instead of common_dispersion if available?
    if dispersion is None:
        dispersion = self.common_dispersion

    return aveLogCPM(self.counts, lib_size=lib_size, prior_count=prior_count, dispersion=dispersion, weights=self.weights)

def aveLogCPM(y, lib_size=None, offset=None, prior_count=2, dispersion=None, weights=None):
    """
    Compute average log2-cpm for each gene over all samples.
    This measure is designed to be used as the x-axis for all abundance-dependent trend analyses in edgeR.
    It is generally held fixed through an edgeR analysis.
    """
    y = np.asarray(y, order='F')
    if len(y.shape) != 2:
        raise ValueError("y should be a matrix")
    if y.shape[0] == 0:
        return 0

    # Special case when all counts and library sizes are zero
    if _isAllZero(y):
        if (lib_size is None or max(lib_size) == 0) and (offset is None or max(offset) == np.NINF):
            abundance = np.full((y.shape[0],), -np.log(y.shape[0]))
            return (abundance + np.log(1e6)) / np.log(2)

    # Check dispersion
    if dispersion is None:
        dispersion = 0.05
    isna = np.isnan(dispersion)
    if isna.all():
        dispersion = 0.05
    elif isna.any():
        dispersion = np.asanyarray(dispersion, order='F')
        dispersion[isna] = np.nanmean(dispersion)

    dispersion = _compressDispersions(y, dispersion)

    # Check weights
    weights = _compressWeights(y, weights)

    # Check offsets
    offset = _compressOffsets(y, lib_size=lib_size, offset=offset)

    # Check prior counts
    prior_count = _compressPrior(y, prior_count)

    # Retrieve GLM fitting parameters
    maxit = signature(mglmOneGroup).parameters['maxit'].default
    tol = signature(mglmOneGroup).parameters['tol'].default

    # Calling the C++ code
    return cxx_ave_log_cpm(y, offset, prior_count, dispersion, weights, maxit, tol)