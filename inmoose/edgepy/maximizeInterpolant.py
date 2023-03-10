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

# This file is based on the file 'R/maximizeInterpolant.R' of the Bioconductor edgeR package (version 3.38.4).


import numpy as np

from .edgepy_cpp import cxx_maximize_interpolant

def maximizeInterpolant(x,y):
    """
    This function takes an ordered set of spline points and a likelihood matrix where each row corresponds to a tag and each column corresponds to a spline point.
    It then calculates the position at which the maximum interpolated likelihood occurs for each by solving the derivative of the spline function.
    """
    x = np.asarray(x, order='F', dtype='double')
    y = np.asarray(y, order='F', dtype='double')
    if len(y.shape) != 2:
        raise ValueError("y is not a matrix: cannot perform interpolation")
    if len(x) != y.shape[1]:
        raise ValueError("number of columns must equal number of spline points")
    if not np.array_equal(np.unique(x), x):
        raise ValueError("spline points must be unique and sorted")

    # Performing some type checking
    out = cxx_maximize_interpolant(x, y)
    return out
