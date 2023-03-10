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

# This file is based on the file 'R/addPriorCount.R' of the Bioconductor edgeR package (version 3.38.4).


import numpy as np
from .makeCompressedMatrix import _compressOffsets, _compressPrior, makeCompressedMatrix
from .edgepy_cpp import cxx_add_prior_count

def addPriorCount(y, lib_size=None, offset=None, prior_count=1):
    """
    Add library size-adjusted prior counts to values of y.
    Also add twice the adjusted prior to th library sizes,
    which are provided as log-transformed values in `offset`.
    """
    # Check y
    y = np.asarray(y, order='F')
    if not np.issubdtype(y.dtype, np.number):
        raise ValueError("count matrix must be numeric")

    # Check prior_count
    prior_count = _compressPrior(y, prior_count)

    # Check lib_size and offset
    # If offsets are provided, they must have a similar average to log(lib_size)
    # for the results to be meaningful as logCPM values
    offset = _compressOffsets(y, lib_size=lib_size, offset=offset)

    # Adding the prior count
    (out_y, out_offset) = cxx_add_prior_count(y, offset, prior_count)
    out_offset = makeCompressedMatrix(out_offset, y.shape, byrow=True)
    return (out_y, out_offset)

