# -----------------------------------------------------------------------------
# Copyright (C) 2004-2022 Gordon Smyth, Yifang Hu, Matthew Ritchie, Jeremy Silver, James Wettenhall, Davis McCarthy, Di Wu, Wei Shi, Belinda Phipson, Aaron Lun, Natalie Thorne, Alicia Oshlack, Carolyn de Graaf, Yunshun Chen, Mette Langaas, Egil Ferkingstad, Marcus Davy, Francois Pepin, Dongseok Choi
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

import numpy as np


class MArrayLM:
    """
    A class to store the results of fitting gene-wise linear models to a set of microarrays

    Objects of this class are normally created by :func:`lmFit` and additional
    components are added by :func:`eBayes`.

    Attributes
    ----------
    coefficients : pd.DataFrame
        matrix containing fitted coefficients or contrasts
    stdev_unscaled : pd.DataFrame
        matrix containing unscaled standard deviations of the coefficients or contrasts
    sigma : ndarray
        array containing residual standard deviations for each gene
    df_residual : ndarray
        array containing residual degrees of freedom for each gene
    Amean : ndarray, optional
        array containing the average log-intensity for each probe over all the
        arrays in the original linear model fit. Note that this vector does not
        change when a contrast is applied to the fit using
        :func:`contrasts_fit`.
    genes : pd.DataFrame, optional
        data frame containing probe annotation
    design : patsy.DesignMatrix, optional
        design matrix
    cov_coefficients : pd.DataFrame, optional
        matrix giving the unscaled covariance matrix of the estimable coefficients
    contrasts : pd.DataFrame, optional
        matrix defining contrasts of coefficients for which results are desired

    s2_prior : ndarray, optional
        single value or array giving empirical Bayes estimated prior value for
        residual variances
    df_prior : ndarray, optional
        value or vector giving empirical Bayes estimated degrees of freedom
        associated with :code:`s2_prior` for each gene
    df_total : ndarray, optional
        array giving total degrees of freedom used for each gene, usually equal
        to :code:`df_prior + df_residual`
    s2_post : ndarray, optional
        array giving posterior residual variances
    var_prior : ndarray, optional
        array giving empirical Bayes estimated prior variance for each true coefficient
    F : ndarray, optional
        array giving moderated *F*-statistics for testing all contrasts equal to zero
    F_p_value : ndarray, optional
        array giving *p*-value corresponding to :code:`F_stat`
    t : pd.DataFrame, optional
        matrix containing empirical Bayes *t*-statistics
    p_value : pd.DataFrame, optional
        matrix of two-sided *p*-values corresponding to the *t*-statistics
    lods : pd.DataFrame, optional
        matrix giving the log-odds of differential expression (on the natural log scale)
    """

    def __init__(self, coefficients, stdev_unscaled, sigma, df_residual, cov_coef):
        self.coefficients = coefficients
        self.stdev_unscaled = stdev_unscaled
        self.sigma = sigma
        self.df_residual = df_residual
        self.cov_coefficients = cov_coef

        self.Amean = None
        self.genes = None
        self.design = None
        self.contrasts = None
        self.s2_prior = None
        self.df_prior = None
        self.df_total = None
        self.s2_post = None
        self.var_prior = None
        self.F = None
        self.F_p_value = None
        self.t = None
        self.p_value = None
        self.lods = None

    def __getitem__(self, idx):
        row_idx, col_idx = idx

        res = MArrayLM(None, None, None, None, None)
        res.coefficients = self.coefficients.loc[row_idx, col_idx]
        res.stdev_unscaled = self.stdev_unscaled.loc[row_idx, col_idx]
        if self.sigma is not None:
            res.sigma = self.sigma[row_idx]
        if self.df_residual is not None:
            res.df_residual = self.df_residual[row_idx]
        if self.cov_coefficients is not None:
            res.cov_coefficients = self.cov_coefficients.loc[col_idx, col_idx]

        if self.Amean is not None:
            res.Amean = self.Amean[row_idx]
        if self.genes is not None:
            res.genes = self.genes.loc[row_idx]
        res.design = self.design
        if self.contrasts is not None:
            res.contrasts = self.contrasts[col_idx]
        if self.s2_prior is not None:
            if not isinstance(self.s2_prior, np.ndarray):
                res.s2_prior = self.s2_prior
            else:
                res.s2_prior = self.s2_prior[row_idx]
        if self.df_prior is not None:
            if not isinstance(self.df_prior, np.ndarray):
                res.df_prior = self.df_prior
            else:
                res.df_prior = self.df_prior[row_idx]
        if self.df_total is not None:
            res.df_total = self.df_total[row_idx]
        if self.s2_post is not None:
            res.s2_post = self.s2_post[row_idx]
        if self.var_prior is not None:
            res.var_prior = self.var_prior[row_idx]
        if self.F is not None:
            res.F = self.F[row_idx]
        if self.F_p_value is not None:
            res.F_p_value = self.F_p_value[row_idx]
        if self.t is not None:
            res.t = self.t.loc[row_idx, col_idx]
        if self.p_value is not None:
            res.p_value = self.p_value.loc[row_idx, col_idx]
        if self.lods is not None:
            res.lods = self.lods.loc[row_idx, col_idx]

        return res
