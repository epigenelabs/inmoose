# -----------------------------------------------------------------------------
# Copyright (C) 2022-2023 M. Colange

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

from scipy.stats import nbinom


def rnbinom(n, size, mu, seed=None):
    """mimic R rnbinom function, to draw samples from a Negative Binomial distribution.

    The (:math:`size`, :math:`p`) parameterization used in R is the same as in scipy.stats:
    :math:`p = 1 / (1 + \mu/size) = size / (size + \mu)`.

    Arguments
    ---------
    n : int or tuple of ints
        shape of the output. If n = (n1, n2, ..., np) then n1*n2*...*np random samples are drawn.
    size : float or array-like
        size parameter of the Negative Binomial distribution.
        all values must be positive
    mu : float or array-like
        mean parameter of the Negative Binomial distribution
        all values must be positive
    seed : int, optional
        pass a seed to the underlying RNG. If `None`, then the RNG is seeded using unpredictable entropy from the system.
        See the documentation of scipy.stats about RNG seeding for more details.
    """
    p = size / (size + mu)
    return nbinom(size, p).rvs(n, random_state=seed)
