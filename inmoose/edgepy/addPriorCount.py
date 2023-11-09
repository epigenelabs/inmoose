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

# This file is based on the file 'R/addPriorCount.R' of the Bioconductor edgeR package (version 3.38.4).


import numpy as np
from .makeCompressedMatrix import _compressOffsets, _compressPrior, makeCompressedMatrix
from .edgepy_cpp import cxx_add_prior_count


def addPriorCount(y, lib_size=None, offset=None, prior_count=1):
    """
    Add a library size-adjusted prior count to each observation.

    This function adds a positive prior count to each observation, often useful
    to avoid zeroes during calculation of log-values. For example,
    :func:`predFC` will call this function to calculate shrunken log-fold
    changes. :func:`aveLogCPM` and :func:`cpm` also use the same underlying
    code to calculate (average) log-counts per million.

    The actual value added to the counts for each library is scaled according
    to the library size. This ensures that the relatives contribution of the
    prior is the same for each library. Otherwise, a fixed prior would have
    little effect on a large library, but a big effect on a small library.

    The library sizes are also modified, with twice the scales prior being
    added to the library size for each library. To understand the motication
    for this, consider that each observation is, effectively, a proportion of
    the total count in the library. The addition scheme implemented here
    represents an empirical logistic transform and ensures that the proportion
    can never be zero or one.

    If :code:`offset` is supplied, this is used in favor of :code:`lib_size`,
    where :code:`exp(offset)` is defined as the vector/matrix of library sizes.
    If an offset matrix is supplied, this will lead to gene-specific scaling of
    the prior as described above.

    Most use cases of this function will involve supplying a constant value to
    :code:`prior_count` for all genes. However, it is also possible to use
    gene-specific values by supplying a vector of length equal to the number of
    rows in :code:`y`.

    Arguments
    ---------
    y : matrix
        a numeric count matrix, with rows corresponding to genes and columns to
        libraries
    lib_size : array, optional
        a numeric vector of library sizes
    offset : array, optional
        a numeric vector or matrix of offsets
    prior_count : float or array
        a constant or gene-specific vector of prior counts to be added genes

    Returns
    -------
    matrix
        matrix of counts with the added priors
    CompressedMatrix
        the log-transformed modified library sizes
    """

    # Check y
    y = np.asarray(y, order="F")
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
