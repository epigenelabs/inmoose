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

# This file is based on the file 'R/glmfit.R' of the Bioconductor edgeR package (version 3.38.4).


import numpy as np

from ..utils import asfactor
from .DGEGLM import DGEGLM
from .makeCompressedMatrix import _compressDispersions, _compressOffsets
from .mglmLevenberg import mglmLevenberg
from .mglmOneWay import designAsFactor, mglmOneWay
from .nbinomDeviance import nbinomDeviance
from .predFC import predFC

from patsy import dmatrix

def glmFit_DGEList(self, design=None, dispersion=None, prior_count=0.125, start=None):
    # The design matrix defaults to the oneway layout defined by self.samples.group
    # If there is only one group, then the design matrix is left None so that a matrix with a single intercept column will be set later by glmFit.
    if design is None:
        design = self.design
        if design is None:
            group = asfactor(self.samples.group).droplevels()
            if group.nlevels() > 1:
                design = dmatrix("~C(self.samples.group)")

    if dispersion is None:
        dispersion = self.getDispersion()
    if dispersion is None:
        raise ValueError("No dispersion values found in DGEList object")
    offset = self.getOffset()
    if self.AveLogCPM is None:
        self.AveLogCPM = self.aveLogCPM()

    fit = glmFit(y=self.counts, design=design, dispersion=dispersion, offset=offset, lib_size=None, weights=self.weights, prior_count=prior_count, start=start)

    fit.samples = self.samples
    fit.genes = self.genes
    fit.prior_df = self.prior_df
    fit.AveLogCPM = self.AveLogCPM
    return fit

def glmFit(y, design=None, dispersion=None, offset=None, lib_size=None, weights=None, prior_count=0.125, start=None):
    """
    Fit negative binomial generalized linear model for each transcript to a series of digital expression libraries
    """
    # Check y
    y = np.asarray(y, order='F')
    (ntag,nlib) = y.shape

    # Check design
    if design is None:
        design = np.ones(shape=(nlib, 1), order='F')
    else:
        design = np.asarray(design, order='F')
        if design.shape[0] != nlib:
            raise ValueError("design should have as many rows as y has columns")
        if np.linalg.matrix_rank(design) < design.shape[1]:
            raise ValueError("Design matrix is not full rank. Some coefficients are not estimable")

    # Check dispersion
    if dispersion is None:
        raise ValueError("No dispersion values provided")
    dispersion = np.asanyarray(dispersion, order='F')
    # TODO check dispersion for NaN and non-numeric values
    if dispersion.shape not in [(), (1,), (ntag,), y.shape]:
        raise ValueError("Dimensions of dispersion do not agree with dimensions of y")
    dispersion_mat = _compressDispersions(y, dispersion)

    # Check offset
    if offset is not None:
        # TODO check that offset is numeric
        offset = np.asanyarray(offset, order='F')
        if offset.shape not in [(), (1,), (nlib,), y.shape]:
            raise ValueError("Dimensions of offset do not agree with dimensions of y")

    # Check lib_size
    if lib_size is not None:
        # TODO check that lib_size is numeric
        lib_size = np.asarray(lib_size, order='F')
        if lib_size.shape not in [(), (1,), (nlib,)]:
            raise ValueError("lib_size has wrong length, should agree with ncol(y)")

    # Consolidate lib_size and offset into a compressed matrix
    offset = _compressOffsets(y=y, lib_size=lib_size, offset=offset)

    # weights are checked in lower-level functions

    # Fit the tagwise GLMs
    # If the design is equivalent to a oneway layout, use a shortcut algorithm
    group = designAsFactor(design)
    if group.nlevels() == design.shape[1]:
        (coef, fitted_values) = mglmOneWay(y, design=design, group=group, dispersion=dispersion_mat, offset=offset, weights=weights, coef_start=start)
        deviance = nbinomDeviance(y=y, mean=fitted_values, dispersion=dispersion_mat, weights=weights)
        fit_method = "oneway"
        fit = (coef, fitted_values, deviance, None, None)
    else:
        fit = mglmLevenberg(y, design=design, dispersion=dispersion_mat, offset=offset, weights=weights, coef_start=start, maxit=250)
        fit_method = "levenberg"

    # Prepare output
    fit = DGEGLM(fit)
    fit.counts = y
    fit.method = fit_method
    if prior_count > 0:
        fit.unshrunk_coefficients = fit.coefficients
        fit.coefficients = predFC(y, design, offset=offset, dispersion=dispersion_mat, prior_count=prior_count, weights=weights) * np.log(2)

    # FIXME (from original R source) we are not allowing missing values, so df.residual must be same for all tags
    fit.df_residual = np.full(ntag, nlib-design.shape[1])
    fit.design = design
    fit.offset = offset
    fit.dispersion = dispersion
    fit.weights = weights
    fit.prior_count = prior_count
    return fit
