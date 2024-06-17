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

# This file is based on the file 'R/validDGEList.R' of the Bioconductor edgeR package (version 3.38.4).


import numpy as np

from ..utils import gl


def validDGEList(y):
    """
    Check for standard components of DGEList object

    NB: Modifies :code:`y` in place.

    Arguments
    ---------
    y : DGEList
        the object to check for validity

    Returns
    -------
    DGEList
        the input :code:`y`, with missing components added
    """
    if y.counts is None:
        raise RuntimeError("No count matrix")
    nlib = y.counts.shape[1]
    if (y.samples.group.values == None).any():  # noqa: E711
        y.samples.group = gl(1, nlib)
    if (y.samples.lib_size.values == None).any():  # noqa: E711
        y.samples.lib_size = y.counts.sum(axis=0)
    if (y.samples.norm_factors.values == None).any():  # noqa: E711
        y.samples.norm_factors = np.ones(nlib)

    return y
