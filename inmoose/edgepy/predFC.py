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

# This file is based on the file 'R/predFC.R' of the Bioconductor edgeR package (version 3.38.4).


import numpy as np
from warnings import warn

from .addPriorCount import addPriorCount

def predFC_DGEList(self, design, prior_count=0.125, offset=None, dispersion=None, weights=None):
    if offset is None:
        offset = self.getOffset()
    if dispersion is None:
        dispersion = self.getDispersion()
    if dispersion is None:
        dispersion = 0
        warn("dispersion set to zero")

    return predFC(y=self.counts, design=design, prior_count=prior_count, offset=offset, dispersion=dispersion, weights=weights)

def predFC(y, design, prior_count=0.125, offset=None, dispersion=0, weights=None):
    """
    Shrink log-fold-changes towards zero by augmenting data counts
    """
    from .glmFit import glmFit
    # Add prior counts in proportion to library size
    (out_y, out_offset) = addPriorCount(y, offset=offset, prior_count=prior_count)

    # Check design
    design = np.asarray(design, order='F')

    # Return matrix of coefficients on log2 scale
    g = glmFit(out_y, design, offset=out_offset, dispersion=dispersion, prior_count=0, weights=weights)
    return g.coefficients / np.log(2)
