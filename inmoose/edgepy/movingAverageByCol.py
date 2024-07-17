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

# This file is based on the file 'R/movingAverageByCol.R' of the Bioconductor edgeR package (version 3.38.4).


from math import ceil, floor

import numpy as np

from ..utils import LOGGER


def movingAverageByCol(x, width=5, full_length=True):
    """
    Moving average smoother for matrix columns.

    Arguments
    ---------
    x : array_like
        numeric matrix
    width : int
        width of window of rows to be averaged
    full_length : bool
        whether the output should have the same number of rows as the input

    Returns
    -------
    ndarray
        numeric matrix with smoothed values. If :code:`full_length = True`, of
        same shape as :code:`x`. If :code:`full_length = False`, has
        :code:`width-1` fewer rows than :code:`x`.
    """
    x = np.asanyarray(x)
    width = int(width)
    if width <= 1:
        return x
    (n, m) = x.shape
    if width > n:
        width = n
        LOGGER.warning("reducing moving average width to x.shape[0]")

    if full_length:
        half1 = ceil(width / 2)
        half2 = floor(width / 2)
        x = np.vstack([np.zeros((half1, m)), x, np.zeros((half2, m))])
    else:
        if width == n:
            return x.mean(axis=0).reshape((1, m))
        x = np.vstack([np.zeros((1, m)), x])

    n2 = x.shape[0]
    x = x.cumsum(axis=0)
    x = x[width:n2] - x[0 : (n2 - width)]
    n3 = x.shape[0]
    w = np.full((n3,), width)
    if full_length:
        if half1 > 1:
            w[0 : half1 - 1] = np.array([width - i for i in range(half1 - 1, 0, -1)])
        w[(n3 - half2) : n3] = np.array([width - i for i in range(1, half2 + 1)])
    return x / w.reshape((n3, 1))
