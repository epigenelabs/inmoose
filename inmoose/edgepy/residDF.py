# -----------------------------------------------------------------------------
# Copyright (C) 2008-2022 Yunshun Chen, Aaron TL Lun, Davis J McCarthy, Matthew E Ritchie, Belinda Phipson, Yifang Hu, Xiaobei Zhou, Mark D Robinson, Gordon K Smyth
# Copyright (C) 2024 Maximilien Colange

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

# This file is based on the file 'R/residDF.R' of the Bioconductor edgeR package (version 3.38.4).

import numpy as np


def _comboGroups(truths):
    """
    Function that returns a list of lists of indices, where each vector refers to the rows with the same combination of True/False values in :code:`truths`.

    Arguments
    ---------
    truths : array_like
        Boolean matrix

    Returns
    -------
    list of lists of ints
        each element of this list is the list of row indices of :code:`truths`
        that contain the same combination of :code:`True`/:code:`False`
    """

    d = {}
    for i in range(truths.shape[0]):
        k = tuple(truths[i, :])
        if k not in d:
            d[k] = []
        d[k].append(i)

    return [v for v in d.values()]


def _residDF(zero, design):
    """
    Effective residual degrees of freedom after adjusting for exact zeros

    Arguments
    ---------
    zero : array_like
        matrix of boolean indicating the zero counts
    design : array_like
        design matrix

    Returns
    -------
    ndarray
        effective residual degrees of freedom, one element per row in :code:`zero`
    """
    nlib = zero.shape[1]
    ncoef = design.shape[1]
    nzero = zero.sum(axis=1)

    # Default is no zero
    DF = np.full(len(nzero), nlib - ncoef)

    # All zero case
    DF[nzero == nlib] = 0

    # Anything in between?
    somezero = (nzero > 0) & (nzero < nlib)
    if somezero.any():
        zero2 = zero.loc[somezero, :].values
        groupings = _comboGroups(zero2)

        # Identifying the true residual d.f. for each of these rows.
        DF2 = nlib - nzero[somezero]
        for i in groupings:
            zeroi = zero2[i[0], :]
            DF2.iloc[i] -= np.linalg.matrix_rank(design[~zeroi, :])
        DF2 = np.maximum(DF2, 0)
        DF[somezero] = DF2

    return DF
