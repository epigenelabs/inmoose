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

# This file is based on the file 'R/splitIntoGroups.R' of the Bioconductor edgeR package (version 3.38.4).


import numpy as np

from ..utils import asfactor


def splitIntoGroups_DGEList(self):
    """
    Split the counts according to group

    See also
    --------
    splitIntoGroups

    Arguments
    ---------
    self : DGEList
        DGEList object, from which the matrix of counts is taken
    group : array_like or :obj:`Factor`
        vector or factor giving the experimental group/condition for each
        library

    Returns
    -------
    list
        list in which each element is a matrix of counts for an individual group.
    """
    group = self.samples["group"]
    return splitIntoGroups(self.counts, group=group)


def splitIntoGroups(y, group=None):
    """
    Split the counts according to group

    Split the counts from a matrix of counts according to a group, creating a
    list where each element consists of a numeric matrix of counts for a
    particular experimental group.

    Arguments
    ---------
    y : array_like
        matrix of counts
    group : array_like or :obj:`Factor`
        vector or factor giving the experimental group/condition for each
        library

    Returns
    -------
    list
        list in which each element is a matrix of counts for an individual group.
    """
    # Check y
    (ntags, nlibs) = y.shape

    # Check group
    if group is None:
        group = np.ones(nlibs)
    if len(group) != nlibs:
        raise ValueError("Incorrect length of group.")
    group = asfactor(group).droplevels()

    out = [y.T[group == i].T for i in np.unique(group)]
    return out
