# distutils: language = c++

cimport cython
from scipy.special cimport cython_special as sp

@cython.ufunc
cdef double nbinom_logpmf(double x, double n, double p):
    coeff = sp.gammaln(n+x) - sp.gammaln(x+1) - sp.gammaln(n)
    return coeff + sp.xlogy(n, p) + sp.xlog1py(x, -p)

