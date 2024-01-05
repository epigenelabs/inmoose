# distutils: language = c++
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

@cython.ufunc
cdef double nbinom_logpmf(double x, double n, double p):
    coeff = sp.gammaln(n+x) - sp.gammaln(x+1) - sp.gammaln(n)
    return coeff + sp.xlogy(n, p) + sp.xlog1py(x, -p)

