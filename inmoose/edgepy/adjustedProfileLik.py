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

# This file is based on the file 'R/adjustedProfileLik.R' of the Bioconductor edgeR package (version 3.38.4).


import numpy as np
from .makeCompressedMatrix import _compressOffsets, _compressDispersions, _compressWeights
from .glmFit import glmFit
from .edgepy_cpp import cxx_compute_apl

def adjustedProfileLik(dispersion, y, design, offset, weights=None, adjust=True, start=None, get_coef=False):
    """
    Tagwise Cox-Reid adjusted log-likelihoods for the dispersion.
    Dispersion can be a scalar or a tagwise vector.
    Computationally, dispersion can also be a matrix, but the APL is still computed tagwise.
    y is a matrix: rows are genes/tags/transcripts, columns are samples/libraries.
    offset is a matrix of the same dimension as y.
    """
    # Checking counts
    y = np.asarray(y, order='F')

    # Checking offsets
    offset = _compressOffsets(y, offset=offset)

    # Checking dispersion
    dispersion = _compressDispersions(y, dispersion)

    # Checking weights
    weights = _compressWeights(y, weights)

    # Fit tagwise linear models
    fit = glmFit(y, design=design, dispersion=dispersion, offset=offset, prior_count=0, weights=weights, start=start)
    mu = fit.fitted_values
    assert mu.dtype == np.dtype('double')
    assert mu.flags.f_contiguous

    # Compute adjusted log-likelihood
    apl = cxx_compute_apl(y, mu, dispersion, weights, adjust, design)

    # Deciding what to return
    if get_coef:
        # TODO
        raise RuntimeError("unimplemented")
    else:
        return apl
