# -----------------------------------------------------------------------------
# Copyright (C) 2008-2022 Yunshun Chen, Aaron TL Lun, Davis J McCarthy, Matthew E Ritchie, Belinda Phipson, Yifang Hu, Xiaobei Zhou, Mark D Robinson, Gordon K Smyth
# Copyright (C) 2022-2025 Maximilien Colange

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

# This file is based on the file 'R/maximizeInterpolant.R' of the Bioconductor edgeR package (version 3.38.4).


import numpy as np

from .edgepy_cpp import maximize_interpolant


def maximizeInterpolant(x, y):
    """
    Maximize a function given a table of values by spline interpolation.

    Calculate the cubic spline interpolant for each row with the method of
    Forsythe et al. (1977) [1]_, then calculate the derivatives of the spline
    segments adjacent to the input with the maximum function value. This allows
    identification of the maximum of the interpolating spline.

    Arguments
    ---------
    x : array_like
        vector of inputs for the function
    y : array_like
        matrix of function values at the values of :code:`x`. Columns correspond
        to :code:`x` values and each row corresponds to a different function to
        be maximized.

    Returns
    -------
    ndarray
        vector of input values at which the function maxima occur

    References
    ----------
    .. [1] G. E. Forsythe, M. A. Malcolm, C. B. Moler. 1977. Computer Methods
       for Mathematical Computations. Prentice-Hall.
    """
    x = np.asarray(x, order="F", dtype="double")
    y = np.asarray(y, order="F", dtype="double")
    if y.ndim != 2:
        raise ValueError("y is not a matrix: cannot perform interpolation")
    if len(x) != y.shape[1]:
        raise ValueError("number of columns must equal number of spline points")
    if not np.array_equal(np.unique(x), x):
        raise ValueError("spline points must be unique and sorted")

    # Performing some type checking
    out = maximize_interpolant(x, y)
    return out
