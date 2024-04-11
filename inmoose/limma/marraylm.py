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


class MArrayLM:
    """
    A class to store the results of fitting gene-wise linear models to a set of microarrays

    Objects of this class are normally created by :func:`lmFit` and additional
    components are added by :func:`eBayes`.

    Attributes
    ----------
    coefficients : ndarray
        matrix containing fitted coefficients or contrasts
    stdev_unscaled : ndarray
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
        data frame containig probe annotation
    design : patsy.DesignMatrix, optional
        design matrix
    cov_coefficients : ndarray, optional
        matrix giving the unscaled covariance matrix of the estimable coefficients
    contrasts : ndarray, optional
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
    t : ndarray, optional
        matrix containing empirical Bayes *t*-statistics
    """

    def __init__(self, coefficients, stdev_unscaled, sigma, df_residual, cov_coef):
        self.coefficients = coefficients
        self.stdev_unscaled = stdev_unscaled
        self.sigma = sigma
        self.df_residual = df_residual
        self.cov_coefficients = cov_coef
