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

# This file is based on the file 'R/splitIntoGroups.R' of the Bioconductor edgeR package (version 3.38.4).


from .dropEmptyLevels import dropEmptyLevels
import numpy as np

def splitIntoGroups_DGEList(self):
    group = self.samples.group
    return splitIntoGroups(self.counts, group=group)

def splitIntoGroups(y, group=None):
    # Check y
    (ntags, nlibs) = y.shape

    # Check group
    if group is None:
        group = np.ones(nlibs)
    if len(group) != nlibs:
        raise ValueError("Incorrect length of group.")
    group = dropEmptyLevels(group)

    out = [y.T[group == i].T for i in np.unique(group)]
    return out
