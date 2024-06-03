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


class DGEGLM(object):
    """
    A simple class to store results of a GLM fit to each gene in  DGE dataset.

    Attributes
    ----------
    coefficients : array_like
        matrix containing the coefficients computed from fitting the model defined by
        the design matrix to each gene in the dataset
    deviance : array_like
        vector giving the deviance from the model fit to each gene
    design : array_like
        design matrix for the full model from the likelihood ratio test
    offset : array_like
        offset values to included in the GLM for each gene
    dispersion : array_like
        dispersion parameters used in the negative binomial GLM for each gene
    weights : array_like
        matrix of weights used in the GLM fitting for each gene
    fitted_values : array_like
        the fitted (expected) values from the GLM for each gene
    AveLogCPM : array_like
        average log2 counts per million for each gene
    """

    from .aveLogCPM import aveLogCPM_DGEGLM as aveLogCPM

    def __init__(self, fit):
        # fit is a 5-tuple
        (coefficients, fitted_values, deviance, iter, failed) = fit
        self.coefficients = coefficients
        self.fitted_values = fitted_values
        self.deviance = deviance
        self.iter = iter
        self.failed = failed

        self.coeff_SE = None
        self.counts = None
        self.design = None
        self.offset = None
        self.dispersion = None
        self.weights = None
        self.prior_count = None
        self.unshrunk_coefficients = None
        self.method = None
        self.AveLogCPM = None
