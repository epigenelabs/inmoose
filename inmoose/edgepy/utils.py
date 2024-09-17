# -----------------------------------------------------------------------------
# Copyright (C) 2008-2022 Yunshun Chen, Aaron TL Lun, Davis J McCarthy, Matthew E Ritchie, Belinda Phipson, Yifang Hu, Xiaobei Zhou, Mark D Robinson, Gordon K Smyth
# Copyright (C) 2022-2024 Maximilien Colange

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

# This file is based on the file 'R/DGEList.R' of the Bioconductor edgeR package (version 3.38.4).


import numpy as np
import pandas as pd


def _isAllZero(y):
    """
    Check for all-zero, negative, Nan and infinite counts

    This function check if :code:`y` is all zero, and raises an error if it
    contains negative, NaN or infinite values.

    Arguments
    ---------
    y : array_like
        matrix of counts

    Returns
    -------
    bool
        whether :code:`y` only contains zeroes
    """
    if isinstance(y, pd.DataFrame):
        y = y.values
    if len(y) == 0:
        return False
    check_range = (np.amin(y), np.nanmax(y))
    if np.isnan(check_range[0]):
        raise ValueError("NaN counts are not allowed")
    if check_range[0] < 0:
        raise ValueError("negative counts are not allowed")
    if np.isinf(check_range[1]):
        raise ValueError("infinite counts are not allowed")
    return check_range[1] == 0
