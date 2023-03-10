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

# This file is based on the file 'R/mglmOneGroup.R' of the Bioconductor edgeR package (version 3.38.4).


import numpy as np

from .makeCompressedMatrix import _compressDispersions, _compressOffsets, _compressWeights
from .utils import _isAllZero
from .edgepy_cpp import cxx_fit_one_group

def mglmOneGroup(y, dispersion=0, offset=0, weights=None, coef_start=None, maxit=50, tol=1e-10, verbose=False):
    """
    Fit single-group negative-binomial glm
    """
    # Check y
    # TODO check that y is a matrix and numeric
    _isAllZero(y)

    # Check dispersion
    dispersion = _compressDispersions(y, dispersion)

    # Check offset
    offset = _compressOffsets(y, offset=offset)

    # Check starting values
    if coef_start is None:
        coef_start = np.NaN
    coef_start = np.full((y.shape[0],), coef_start, dtype='double', order='F')

    # Check weights
    weights = _compressWeights(y, weights)

    # Fisher scoring iteration
    output = cxx_fit_one_group(y, offset, dispersion, weights, maxit, tol, coef_start)

    # Convergence achieved for all tags?
    if verbose and np.count_nonzero(output[1]) > 0:
        warn("max iterations exceeded for ", np.count_nonzero(output[1]), "tags")

    return output[0]

