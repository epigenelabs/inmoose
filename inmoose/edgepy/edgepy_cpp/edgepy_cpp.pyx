# distutils: language = c++
#-----------------------------------------------------------------------------
# Copyright (C) 2008-2022 Yunshun Chen, Aaron TL Lun, Davis J McCarthy, Matthew E Ritchie, Belinda Phipson, Yifang Hu, Xiaobei Zhou, Mark D Robinson, Gordon K Smyth
# Copyright (C) 2022-2024 Maximilien Colange

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

# This file contains Cython ports of the original C++ code from various files of
# the Bioconductor edgeR package (version 3.38.4):
# - 'src/nbdev.cpp' and 'src/R_compute_nbdev.cpp' (function
#   `compute_unit_nb_deviance`)
# - 'src/R_compute_apl.cpp' and 'src/adj_coxreid.cpp' (functions `compute_apl`
#   and `acr_compute`)
# - 'src/glm_one_group.cpp' (function `glm_one_group_cython`)
# - 'R/q2qnbinom.R' (function _q2qnbinom)

import numpy as np
cimport cython
from libcpp.cmath cimport abs, log, isfinite, exp, isnan
from numpy.math cimport INFINITY

from libc.math cimport sqrt
from scipy.special cimport cython_special as sp
from scipy.special.cython_special cimport gammaln as lgamma
from scipy.linalg.lapack import get_lapack_funcs


cdef public ndarray[double, ndim=1] vector2ndarray "vector2ndarray"(const vector.vector[double]& data):
    return np.asarray(data)


ctypedef fused count_type:
    long
    double


cdef double low_value = 1e-10
cdef double log_low_value = log(low_value)
cdef double mildly_low_value = 1e-8


@cython.ufunc
@cython.cdivision(True)
cdef double compute_unit_nb_deviance(double y, double mu, double phi):
    """
    Calculate the deviance of a negative binomial fit

    Note the protection for very large mu*phi (where we use a Gamma instead) or
    very small mu*phi (where we use a Poisson instead). This approximation
    protects against numerical instability introduced by subtracting a very
    large log value in (log mu) with another very large logarithm (log mu+1/phi).
    We need to consider the phi as the approximation is only good when the
    product is very big or very small.

    Arguments
    ---------
    y : array_like
        counts matrix
    mu : array_like
        expected means matrix (broadcastable to the shape of :code:`y`)
    phi : array_like
        dispersion matrix (broadcastable to the shape of :code:`y`)

    Returns
    -------
    ndarray
        matrix of deviances (same shape as :code:`y`)
    """
    # add a small value to protect against zero during division and log
    y = y + mildly_low_value
    mu = mu + mildly_low_value

    # Calculating the deviance using either the Poisson (small phi*mu), the
    # Gamma (large phi*mu) or NB (everything else). Some additional work is
    # put in to make the transitiosn between families smooth
    resid = y - mu
    product = mu * phi

    if phi < 1e-4:
        return 2 * (y * log(y/mu) - resid - 0.5*resid*resid*phi*(1+phi*(2/3*resid-y)))
    elif product > 1e6:
        return 2 * (resid/mu - log(y/mu)) * mu/(1+product)
    else:
        return 2 * (y * log(y/mu) + (y + 1/phi) * log((mu + 1/phi)/(y + 1/phi)))

@cython.ufunc
cdef double _q2qnbinom(double x, double input_mean, double output_mean, double dispersion):
    """
    Interpolated quantile to quantile mapping between negative-binomial distributions with the same dispersion but different means.

    This is the low-level Cythonized function called by :func:`q2qnbinom`.

    See also
    --------
    q2qnbinom

    Arguments
    ---------
    x : array_like
        matrix of counts
    input_mean : array_like
        matrix of population means for :code:`x`. If 1-D, then of the same
        length as :code:`x.shape[0]`
    output_mean : array_like
        matrix of population means for the output values. If 1-D, then of the
        same length as :code:`x.shape[0]`
    dispersion : array_like
        scalar, vector or matrix giving negative binomial dispersion values

    Returns
    -------
    ndarray
        matrix of same dimensions as :code:`x`, with :code:`output_mean` as the
        new nominal population mean
    """
    if x < 0:
        raise ValueError("x must be non-negative")
    if input_mean < 0:
        raise ValueError("input_mean must be non-negative")
    if output_mean < 0:
        raise ValueError("output_mean must be non-negative")
    if dispersion < 0:
        raise ValueError("dispersion must be non-negative")

    eps = 1e-14
    if input_mean < eps or output_mean < eps:
        input_mean += 0.25
        output_mean += 0.25
    ri = 1 + dispersion * input_mean
    vi = input_mean * ri
    ro = 1 + dispersion * output_mean
    vo = output_mean * ro

    q1 = output_mean + sqrt(vo / vi) * (x - input_mean)

    if x >= input_mean:
        p2 = sp.gammaincc(input_mean / ri, x/ri)
        q2 = sp.gammainccinv(output_mean / ro, p2) * ro
    else:
        p2 = sp.gammainc(input_mean / ri, x/ri)
        q2 = sp.gammaincinv(output_mean / ro, p2) * ro

    return (q1+q2)/2


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef ndarray compute_apl(const count_type[:,:] y, const double[:,:] means, const double[:,:] disps, const double[:,:] weights, bool adjust, ndarray design):
    """
    Compute adjusted profile log-likelihoods of genewise negative binomial GLMs

    This is the low-level function called by :func:`adjustedProfileLik`.

    Arguments
    ---------
    y : array_like
        matrix of counts
    means : array_like
        matrix of expected means, same shape as :code:`y`
    disps : array_like
        matrix of dispersions
    weights : array_like
        matrix of observation weights, same shape as :code:`y`
    adjust : bool
        whether to use Cox-Reid adjustment
    design : array_like
        the design matrix

    Returns
    -------
    ndarray
        the genewise APL (one element per row in :code:`y`)
    """

    cdef Py_ssize_t tag, lib
    cdef double curmu, cury, curd, r, logmur, adj
    cdef long lwork = -1

    if adjust:
        routine = get_lapack_funcs('sytrf_lwork', dtype=np.double)
        ret = routine(design.shape[1])
        assert len(ret) == 2
        if ret[1] != 0:
            raise ValueError("Internal work array size computation failed: "
                             "%d" % (ret[1],))
        lwork = int(ret[0]+0.5)
        if lwork < 1:
            lwork = 1

    res = np.zeros(y.shape[0])
    cdef double[:] sum_loglike = res

    working_weights_array = np.zeros(y.shape[1])
    cdef double[:] working_weights = working_weights_array

    for tag in range(y.shape[0]):
        for lib in range(y.shape[1]):
            # mean should only be zero if count is zero, where the
            # log-likelihood would then be 0.
            if means[tag,lib] == 0:
                continue

            # each y is assumed to be the average of 'weights' counts, so we
            # convert from averages to the "original sums" in order to compute
            # NB probabilities
            curmu = means[tag,lib] * weights[tag,lib]
            cury = y[tag,lib] * weights[tag,lib]
            curd = disps[tag,lib] / weights[tag,lib]

            # compute the log-likelihood
            r = 1 / curd
            logmur = log(curmu + r)

            if curd > 0:
                sum_loglike[tag] += cury*log(curmu) - cury*logmur + r*log(r) - r*logmur + lgamma(cury+r) - lgamma(cury+1) - lgamma(r)
            else:
                sum_loglike[tag] += cury*log(curmu) - curmu - lgamma(cury+1)

            # adding the Jacobian, to account for the fact that we actually
            # want the log-likelihood of the _scaled_ NB distribution (after
            # dividing the original sum by the weight).
            sum_loglike[tag] += log(weights[tag,lib])

            if adjust:
                # computing W, the matrix of NB working weights
                # this is used to compute the Cox-Reid adjustment factor
                working_weights[lib] = curmu / (1 + curd*curmu)

        if adjust:
            if design.shape[1] == 1:
                adj = working_weights_array.sum()
                adj = 0.5*log(abs(adj))
                sum_loglike[tag] -= adj
            else:
                sum_loglike[tag] -= acr_compute(working_weights, design, lwork)

    return res


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double acr_compute(const double[:] wptr, ndarray design, long lwork):
    """
    Compute the Cox-Reid adjustment factor

    XtWX represents the expected Fisher information. The overall strategy is to
    compute the log-determinant of this matrix, to compute the adjustment factor
    for the likelihood (in order to account for uncertainty in the nuisance
    parameters i.e. the fitted values).

    We want to apply the Cholesky decomposition to the XtWX matrix. However, to
    be safe, we call the routine to do a symmetric indefinite factorization i.e.
    A = LDLt. This guarantees factorization for singular matrices when the
    actual Cholesky decomposition would fail because it would start whining
    about non-positive eigenvectors.

    We then try to compute the determinant of XtWX, using two facts:

    - for triangular matrices, the determinant is the product of the diagonals
    - det(LDL*) = det(L)*det(D)*det(L*)
    - all diagonal elements of L are unity

    Thus, all we need to do is to sum over all log'd diagonal elements in D. We
    then divide by two, because that is just the definition of the Cox-Reid
    adjustment.

    If one of the diagonal elements is zero or NA, we replace it with an
    appropriately small non-zero value. This is valid because the zero elements
    correspond to all-zero column in WX, which in turn only arises when there
    are fitted values of zero, which will be constant at all dispersions. Thus,
    any replacement value will eventually cancel out during interpolation to
    obtain the CRAPLE.

    Note that the scipy routine will also do some pivoting, essentially solving
    PAP* = LDL* for some permutation matrix P. This should not affect anything:
    the determinant of the permutation matrix is either 1 or -1, but it cancels
    out, so det(A) = det(PAP*).

    Further note that the routine can theoretically give block diagonals, but
    this should not occur for positive (semi)definite matrices, which is what
    XtWX should always be.

    Arguments
    ---------
    wptr : array_like
        weights array
    design : array_like
        the design matrix

    Returns
    -------
    float
        the Cox-Reid adjustment factor
    """
    cdef double res = 0
    cdef Py_ssize_t i

    xtwx = (design.T * wptr) @ design

    # BEGIN LDL* decomposition
    solver = get_lapack_funcs('sytrf', dtype=np.double)
    ldu, _, info = solver(xtwx, lwork=lwork, lower=True, overwrite_a=False)
    if info < 0:
        raise ValueError('{} exited with the internal error "illegal value '
                         'in argument number {}". See LAPACK documentation '
                         'for the error codes.'.format('DSYTRF', -info))
    # END LDL* decomposition

    cdef double[:,:] d = ldu

    # log-determinant as sum of the log-diagonals, then halving
    assert d.shape[0] == d.shape[1]
    for i in range(ldu.shape[0]):
        if d[i,i] < low_value or not isfinite(d[i,i]):
            res += log_low_value
        else:
            res += log(d[i,i])

    return 0.5*res


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef (double, bool) glm_one_group_cython(Py_ssize_t nlibs,
                                          const count_type[:] counts, const double[:] offset,
                                          const double[:] disp, const double[:] weights,
                                          long maxit, double tolerance, double cur_beta):
    """
    Simplified fit for negative binomial GLM when the design matrix is one group

    This is the low-level function ultimately called by :func:`fit_one_group`.

    See Also
    --------
    glm_one_group
    fit_one_group

    Arguments
    ---------
    nlibs : Py_ssize_t
        number of libraries
    counts : array_like
        vector of counts for a given gene, of size `nlibs`
    offset : array_like
        vector of offsets for a given gene, of size `nlibs`
    disp : array_like
        vector of dispersions for a given gene, of size `nlibs`
    weights : array_like
        vector of observation weights for a given gene, of size `nlibs`
    maxit : long
        maximum number of Newton-Raphson iterations
    tol : float
        tolerance for convergence in the Newton-Raphson iteration
    cur_beta : float
        initial beta coefficient

    Returns
    -------
    float
        fitted coefficient
    bool
        whether the fit converged
    """

    cdef Py_ssize_t j
    cdef long i
    cdef double totweight, dl, info, mu, cur_val

    # Setting up initial values for beta as the log of the mean of the ratio of
    # counts to offsets. This is the exact solution for the gamma distribution
    # (which is the limit of the NB as the dispersion goes to infinity).
    # However, if cur_beta is not NA, then we assume it is good.
    nonzero = False
    if isnan(cur_beta):
        cur_beta = 0
        totweight = 0
        for j in range(nlibs):
            cur_val = counts[j]
            if cur_val > low_value:
                cur_beta += cur_val / exp(offset[j]) * weights[j]
                nonzero = True
            totweight += weights[j]
        cur_beta = log(cur_beta / totweight)
    else:
        for j in range(nlibs):
            if counts[j] > low_value:
                nonzero = True
                break

    # skipping to a result for all-zero rows
    if not nonzero:
        return (-INFINITY, True)

    # Newton-Raphson iteration to converge to mean
    has_converged = False
    for i in range(maxit):
        dl = 0
        info = 0

        for j in range(nlibs):
            mu = exp(cur_beta + offset[j])
            denom = 1 + mu*disp[j]
            dl += (counts[j] - mu) / denom * weights[j]
            info += mu / denom * weights[j]
        step = dl / info
        cur_beta += step
        if abs(step) < tolerance:
            has_converged = True
            break

    return (cur_beta, has_converged)
