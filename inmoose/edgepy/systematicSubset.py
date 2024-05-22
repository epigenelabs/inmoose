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

# This file is based on the file 'R/systematicSubset.R' of the Bioconductor edgeR package (version 3.38.4).


from math import floor

import numpy as np


def systematicSubset(n, order_by):
    """
    Take a systematic subset of indices stratified by a ranking variable

    Arguments
    ---------
    n : int
        the size of the subset
    order_by : array_like
        vector of the values by which the indices are ordered

    Returns
    -------
    ndarray
        a vector of size :code:`n`
    """
    ntotal = len(order_by)
    sampling_ratio = floor(ntotal / n)
    if sampling_ratio <= 1:
        return np.arange(ntotal)
    i1 = floor(sampling_ratio / 2)
    i = np.arange(i1, ntotal, step=sampling_ratio)
    o = np.argsort(order_by)
    return o[i]
