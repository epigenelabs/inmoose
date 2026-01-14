# distutils: language = c++
# cython: language_level=3
#-----------------------------------------------------------------------------
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

cimport cython
from scipy.special cimport cython_special as sp

cdef double _nbinom_logpmf_c(double x, double n, double p):
    """Internal C function for computing log PMF."""
    cdef double coeff = sp.gammaln(n+x) - sp.gammaln(x+1) - sp.gammaln(n)
    return coeff + sp.xlogy(n, p) + sp.xlog1py(x, -p)

cpdef nbinom_logpmf(x, n, p):
    """Compute log probability mass function of negative binomial distribution.
    
    Arguments
    ---------
    x : float or array-like
        number of successes
    n : float or array-like
        size parameter (number of successes)
    p : float or array-like
        probability of success
    
    Returns
    -------
    float or array
        log probability mass
    """
    cdef double x_val, n_val, p_val
    x_val = float(x)
    n_val = float(n)
    p_val = float(p)
    return _nbinom_logpmf_c(x_val, n_val, p_val)
