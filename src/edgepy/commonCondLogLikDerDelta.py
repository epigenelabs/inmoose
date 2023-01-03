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

# This file is based on the file 'R/commonCondLogLikDerDelta.R' of the Bioconductor edgeR package (version 3.38.4).


from .condLogLikDerDelta import condLogLikDerDelta

def commonCondLogLikDerDelta(delta, y, der=0):
    """
    Calculate the common conditional log-likelihood (i.e. summed over all tags)
    This function is necessary so that minimization can be applied in `estimateCommonDisp`
    """
    l0 = 0
    for x in y:
        l0 += condLogLikDerDelta(delta, y=x, der=der).sum()
    return l0
