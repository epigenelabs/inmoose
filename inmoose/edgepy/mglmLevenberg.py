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

# This file is based on the file 'R/mglmLevenberg.R' of the Bioconductor edgeR package (version 3.38.4).


import numpy as np

from .edgepy_cpp import cxx_get_levenberg_start, cxx_fit_levenberg
from .makeCompressedMatrix import _compressDispersions, _compressOffsets, _compressWeights
from .utils import _isAllZero

def mglmLevenberg(y, design, dispersion=0, offset=0, weights=None, coef_start=None, start_method="null", maxit=200, tol=1e-6):
    """
    Fit genewise negative binomial GLMs with log-link using Levenberg damping to ensure convergence
    """
    # Check arguments
    y = np.asarray(y, order='F')
    (ngenes, nlibs) = y.shape
    if nlibs == 0 or ngenes == 0:
        raise ValueError("no data")

    # Check for negative, NA or non-finite values in the count matrix
    _isAllZero(y)

    # Check the design matrix
    design = np.asarray(design, order='F', dtype='double')
    # TODO check that all entries in design matrix are finite

    # Check dispersions, offsets, and weights
    offset = _compressOffsets(y, offset=offset)
    dispersion = _compressDispersions(y, dispersion)
    weights = _compressWeights(y, weights)

    # Initialize values for the coefficients at reasonable best guess with linear models
    if coef_start is None:
        if start_method not in ["null", "y"]:
            raise ValueError(f"invalid start_method {start_method}")
        beta = cxx_get_levenberg_start(y, offset, dispersion, weights, design, start_method=="null")
    else:
        beta = np.asarray(coef_start, order='F', dtype='double')

    assert beta.shape == (y.shape[0], design.shape[1])
    # Check the arguments and call the C++ method
    output = cxx_fit_levenberg(y, offset, dispersion, weights, design, beta, tol, maxit)

    # Name the output and return it
    # TODO
    return output
