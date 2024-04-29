# -----------------------------------------------------------------------------
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
from statsmodels.regression.linear_model import WLS


class lm_wfit:
    """
    Python equivalent to R :code:`lm.wfit`

    This class is merely a wrapper around :class:`RegressionResults`, the
    return type of :meth:`OLS.fit` and :meth:`WLS.fit`.

    It contains the solution to the equation :math:`A X = B`, where :math:`A`
    is a known matrix, :math:`B` a known vector and :math:`X` the unknown to
    solve for.  The solution is obtained by a weighted least-square-error fit
    based on the QR-decomposition of :math:`A`.

    Arguments
    ---------
    design : array_like
        matrix :math:`A` of the equation
    b : array_like
        offset (vector :math:`B` in the equation)
    w : array_like
        weights to use in the least-square-error fit
    """

    def __init__(self, design, b, w):
        self.fit = WLS(b, design, w).fit(method="qr")
        self._effects = None

    @property
    def coefficients(self):
        """
        ndarray: the value found for :math:`X`
        """
        return self.fit.params

    @property
    def residuals(self):
        """
        ndarray: the residuals, *i.e.* the difference between :math:`A X` and :math:`B`
        """
        return self.fit.resid

    @property
    def fitted_values(self):
        """
        ndarray: the value :math:`A X`
        """
        return self.fit.fittedvalues

    @property
    def effects(self):
        """
        ndarray: vector of orthogonal single-df effects. The first :code:`rank` of them
        correspond to the :code:`coefficients`
        """
        if self._effects is None:
            Q, _ = np.linalg.qr(self.fit.model.wexog, mode="complete")
            self._effects = Q.T @ self.fit.model.wendog
        return self._effects

    @property
    def weights(self):
        """
        ndarray: the weights used in the least-square-error fit
        """
        return self.fit.weights

    @property
    def rank(self):
        """
        int: the column rank of matrix :math:`A`
        """
        return self.fit.model.rank

    @property
    def df_residuals(self):
        """
        ndarray: degrees of freedom of residuals
        """
        return self.fit.model.df_resid

    @property
    def qr(self):
        """
        ndarray, ndarray: the QR decomposition of matrix :math:`A`
        """
        return self.fit.model.exog_Q, self.fit.model.exog_R


class lm_fit(lm_wfit):
    """
    Python equivalent to R :code:`lm.fit`

    This class is merely a wrapper around :class:`RegressionResults`, the
    return type of :meth:`OLS.fit` and :meth:`WLS.fit`.

    It contains the solution to the equation :math:`A X = B`, where :math:`A`
    is a known matrix, :math:`B` a known vector and :math:`X` the unknown to
    solve for.  The solution is obtained by a least-square-error fit based on
    the QR-decomposition of :math:`A`.

    Arguments
    ---------
    design : array_like
        matrix :math:`A` of the equation
    b : array_like
        offset (vector :math:`B` in the equation)
    """

    def __init__(self, design, b):
        super().__init__(design, b, 1.0)
