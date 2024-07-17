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

# original code was contributed to StackOverflow: https://stackoverflow.com/questions/71550468/does-python-have-an-analogue-to-rs-splinesns
# code was improved to better match the R code

# As per StackOverflow terms of service (see https://stackoverflow.com/help/licensing), original code is licensed under CC BY SA 4.0, which is compatible with GPL3 (see https://creativecommons.org/2015/10/08/cc-by-sa-4-0-now-one-way-compatible-with-gplv3/)

import numpy as np
from scipy.interpolate import splev

from ..utils import LOGGER


def spline_design(knots, x, order, derivs=0):
    """
    Evaluate the design matrix for the B-splines defined by :code:`knots` at the values in :code:`x`.

    Arguments
    ---------
    knots : array_like
        vector of knot positions (which will be sorted increasingly if needed).
    x : array_like
        vector of values at which to evaluate the B-spline functions or
        derivatives. The values in x must be between the “inner” knots
        :code:`knots[ord]` and :code:`knots[ length(knots) - (ord-1)]`
    order : int
        a positive integer giving the order of the spline function. This is the
        number of coefficients in each piecewise polynomial segment, thus a
        cubic spline has order 4.
    derivs : array_like
        an integer vector with values between 0 and :code:`ord` - 1,
        conceptually recycled to the length of :code:`x`. The derivative of the
        given order is evaluated at the :code:`x` positions. Defaults to zero.

    Returns
    -------
    ndarray
        a matrix with :code:`len(x)` rows and :code:`len(knots) - ord` columns.
        The :code:`i`'th row of the matrix contains the coefficients of the
        B-splines (or the indicated derivative of the B-splines) defined by the
        knot vector and evaluated at the :code:`i`'th value of :code:`x`. Each
        B-spline is defined by a set of :code:`ord` successive knots so the
        total number of B-splines is :code:`len(knots) - ord`.
    """
    derivs = np.asarray(derivs)
    if derivs.ndim == 0:
        der = np.repeat(derivs, len(x))
    else:
        der = np.zeros(len(x), dtype=int)
        der[: len(derivs)] = derivs
    n_bases = len(knots) - order
    res = np.empty((len(x), n_bases), dtype=float)
    for i in range(n_bases):
        coefs = np.zeros((n_bases,))
        coefs[i] = 1
        for j in range(len(x)):
            res[j, i] = splev(x, (knots, coefs, order - 1), der=der[j])[j]
    return res


class ns:
    """
    Class storing the B-spline basis matrix for a natural cubic spline and info used to generate it.

    Attributes
    ----------
    knots : array_like
        breakpoints that define the spline.
    include_intercept : bool
        whether an intercept is included in the basis
    boundary_knots : array_like
        boundary points at which to impose the natural boundary conditions and
        anchor the B-spline basis.
    basis : ndarray
        a matrix of dimension :code:`(len(x), df)`, where :code:`df =
        len(knots)-1-intercept` if :code:`df` was not supplied.
    """

    def __init__(
        self, x, df=None, knots=None, boundary_knots=None, include_intercept=False
    ):
        """
        Generate the B-spline basis matrix for a natural cubic spline.

        This function intends to provide the same functionality as R splines::ns.

        Arguments
        ---------
        x : array_like
            the predictor variable
        df : int, optional
            degrees of freedom. If :code:`knots` is not specified, then the
            function chooses :code:`df - 1 - intercept` knots at suitably chosen
            quantiles of :code:`x`. If :code:`None`, the number of inner knots is
            set to :code:`len(knots)`.
        knots : array_like, optional
            breakpoints that define the spline. The default is no knots; together
            with the natural boundary conditions this results in a basis for linear
            regression on :code:`x`. Typical values are the mean or median for one
            knot, quantiles for more knots. See also :code:`boundary_knots`.
        include_intercept : bool, optional
            if :code:`True`, an intercept is included in the basis; default is
            :code:`False`.
        boundary_knots : array_like, optional
            boundary points at which to impose the natural boundary conditions and
            anchor the B-spline basis (default the range of the data). If both
            :code:`knots` and :code:`boundary_knots` are supplied, the basis
            parameters do not depend on :code:`x`. Data can extend beyond
            :code:`boundary_knots`.

        Returns
        -------
        ndarray
            a matrix of dimension :code:`(len(x), df)`, where :code:`df =
            len(knots)-1-intercept` if :code:`df` was not supplied.
        """
        self.include_intercept = include_intercept
        x = np.asarray(x)
        if boundary_knots is None:
            boundary_knots = [np.min(x), np.max(x)]
            outside = False
        else:
            boundary_knots = list(np.sort(boundary_knots))
            out_left = x < boundary_knots[0]
            out_right = x > boundary_knots[1]
            outside = out_left | out_right
        self.boundary_knots = boundary_knots

        if df is not None and knots is None:
            nIknots = df - 1 - include_intercept
            if nIknots < 0:
                nIknots = 0
                LOGGER.warning("df was too small, used {1+include_intercept}")

            if nIknots > 0:
                knots = np.linspace(0, 1, num=nIknots + 2)[1:-1]
                knots = np.quantile(x, knots)
        else:
            nIknots = len(knots)
        self.knots = knots

        Aknots = np.sort(np.concatenate((boundary_knots * 4, knots)))

        if np.any(outside):
            basis = np.empty((x.shape[0], nIknots + 4), dtype=float)
            if np.any(out_left):
                k_pivot = boundary_knots[0]
                xl = np.ones((np.sum(out_left), 2))
                xl[:, 1] = x[out_left] - k_pivot
                tt = spline_design(Aknots, [k_pivot, k_pivot], 4, [0, 1])
                basis[out_left, :] = xl @ tt
            if np.any(out_right):
                k_pivot = boundary_knots[1]
                xr = np.ones((np.sum(out_right), 2))
                xr[:, 1] = x[out_right] - k_pivot
                tt = spline_design(Aknots, [k_pivot, k_pivot], 4, [0, 1])
                basis[out_right, :] = xr @ tt
            inside = ~outside
            if np.any(inside):
                basis[inside, :] = spline_design(Aknots, x[inside], 4)
        else:
            basis = spline_design(Aknots, x, 4)

        const = spline_design(Aknots, boundary_knots, 4, [2, 2])

        if include_intercept is False:
            basis = basis[:, 1:]
            const = const[:, 1:]

        qr_const = np.linalg.qr(const.T, mode="complete")[0]
        self.basis = (qr_const.T @ basis.T).T[:, 2:]

    def predict(self, newx):
        """
        Evaluate the spline basis at given values

        Arguments
        ---------
        newx : ndarray
            new predictor variable to regenerate the spline from

        Returns
        -------
        ns
            a new natural spline object, evaluated at the given values
        """
        return ns(
            newx,
            knots=self.knots,
            boundary_knots=self.boundary_knots,
            include_intercept=self.include_intercept,
        )
