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

# This file is based on the file 'R/estimateGLMTagwiseDisp.R' of the Bioconductor edgeR package (version 3.38.4).


import numpy as np

from .aveLogCPM import aveLogCPM
from .dispCoxReidInterpolateTagwise import dispCoxReidInterpolateTagwise

def estimateGLMTagwiseDisp_DGEList(self, design=None, prior_df=10, trend=None, span=None):
    """
    Modifies self in place
    """
    # Find appropriate dispersion
    if trend is None:
        trend = self.trended_dispersion is not None
    if trend:
        dispersion = self.trended_dispersion
        if dispersion is None:
            raise ValueError("No trended_dispersion in data object. Run estimateGLMTrendedDisp first.")
    else:
        dispersion = self.common_dispersion
        if dispersion is None:
            raise ValueError("No common_dispersion found in data object. Run estimateGLMCommonDisp first.")

    if self.AveLogCPM is None:
        self.AveLogCPM = self.aveLogCPM()
    ntags = self.counts.shape[0]

    if span is None:
        if ntags > 10:
            span = (10 / ntags)**0.23
        else:
            span = 1
    self.span = span

    d = estimateGLMTagwiseDisp(self.counts, design=design, offset=self.getOffset(), dispersion=dispersion, trend=trend, span=span, prior_df=prior_df, AveLogCPM=self.AveLogCPM, weights=self.weights)
    self.prior_df = prior_df
    self.tagwise_dispersion = d
    return self

def estimateGLMTagwiseDisp(y, dispersion, design=None, offset=None, prior_df=10, trend=True, span=None, AveLogCPM=None, weights=None):
    # Check y
    y = np.asarray(y, order='F')
    (ntags, nlibs) = y.shape
    if ntags == 0:
        return 0

    # Check design
    if design is None:
        design = np.ones((y.shape[1], 1), order='F')
    else:
        design = np.asarray(design, order='F')

    if design.shape[1] >= y.shape[1]:
        warn("No residual df: setting dispersion to NA")
        return np.full((ntags,), np.nan, order='F')

    # Check offset
    if offset is None:
        offset = np.log(y.sum(axis=0))

    # Check span
    # span can be chosen smaller when ntags is large
    if span is None:
        if ntags > 10:
            span = (10 / ntags)**0.23
        else:
            span = 1

    # Check AveLogCPM
    if AveLogCPM is None:
        AveLogCPM = aveLogCPM(y, offset=offset, weights=weights)

    # Call Cox-Reid grid method
    tagwise_dispersion = dispCoxReidInterpolateTagwise(y, design, offset=offset, dispersion=dispersion, trend=trend, prior_df=prior_df, span=span, AveLogCPM=AveLogCPM, weights=weights)
    return tagwise_dispersion
