# distutils: language = c++
#-----------------------------------------------------------------------------
# Copyright (C) 2008-2022 Yunshun Chen, Aaron TL Lun, Davis J McCarthy, Matthew E Ritchie, Belinda Phipson, Yifang Hu, Xiaobei Zhou, Mark D Robinson, Gordon K Smyth
# Copyright (C) 2022-2025 Maximilien Colange

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
# - 'src/interpolator.cpp' and 'src/interpolator.h' (class `interpolator` and function
#   `quad_solver`)
# - 'src/R_maximize_interpolant.cpp' (function `maximize_interpolant` function)

import numpy as np
cimport cython
from libcpp.cmath cimport abs, log, isfinite, exp, isnan
from libc.math cimport INFINITY
from libc.stdint cimport int64_t

from libc.math cimport sqrt
from scipy.special cimport cython_special as sp
from scipy.special.cython_special cimport gammaln as lgamma
from scipy.linalg.lapack import get_lapack_funcs


cdef public ndarray[double, ndim=1] vector2ndarray "vector2ndarray"(const vector.vector[double]& data):
    return np.asarray(data)


ctypedef fused count_type:
    int64_t
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


cdef struct solution:
    double sol1, sol2
    bool solvable

cdef solution quad_solver(double a, double b, double c):
    """
    Solve quadratic equation ax^2 + bx + c = 0.

    Parameters
    ----------
    a : double
        Coefficient of x^2
    b : double
        Coefficient of x
    c : double
        Constant term

    Returns
    -------
    solution
        Structure containing the solutions and whether they exist
    """
    cdef:
        solution cur_sol
        double back, front

    if a == 0 and b == 0:
        if c == 0:
            cur_sol.solvable = True
        else:
            cur_sol.solvable = False
        return cur_sol

    if a == 0:
        cur_sol.sol1 = -c/b
        cur_sol.sol2 = -c/b
        cur_sol.solvable = True
        return cur_sol

    back = b * b - 4 * a * c
    if back < 0:
        cur_sol.solvable = False
        return cur_sol

    front = -b / (2 * a)
    back = sqrt(back) / (2 * a)
    cur_sol.sol1 = front - back
    cur_sol.sol2 = front + back
    cur_sol.solvable = True
    return cur_sol


cdef class interpolator:
    """
    Class to identify the global maximum in the interpolating function.

    This is a Cython port of the C++ interpolator class from edgeR.

    Attributes
    ----------
    npts : Py_ssize_t
        number of points in cubic spline fit
    b : vector[double]
        vector of coefficients of degree 1 for the cubic spline
    c : vector[double]
        vector of coefficients of degree 2 for the cubic spline
    d : vector[double]
        vector of coefficients of degree 3 for the cubic spline
    """

    cdef:
        Py_ssize_t npts
        vector.vector[double] b, c, d

    def __init__(self, Py_ssize_t n):
        """
        Initialize the interpolator with the number of points.

        Parameters
        ----------
        n : size_t
            Number of points for interpolation (must be at least 2)
        """
        if n < 2:
            raise RuntimeError("must have at least two points for interpolation")
        self.npts = n
        self.b.resize(n)
        self.c.resize(n)
        self.d.resize(n)

    cdef double find_max(self, const double* x, const double* y):
        """
        Find the maximum of the interpolating spline.

        Parameters
        ----------
        x : const double*
            Array of x-coordinates
        y : const double*
            Array of y-coordinates

        Returns
        -------
        double
            The x-coordinate where the maximum occurs
        """
        cdef:
            double maxed = -1
            Py_ssize_t maxed_at = -1
            Py_ssize_t i
            double x_max

            double ld, lc, lb
            double rd, rc, rb
            solution sol_left, sol_right
            double chosen_sol, temp

        # Get initial guess for MLE
        for i in range(self.npts):
            if maxed_at == -1 or y[i] > maxed:
                maxed = y[i]
                maxed_at = i

        x_max = x[maxed_at]

        # Fit the spline
        self._fmm_spline(x, y)

        # Check left segment for maximum
        if maxed_at > 0:
            ld = self.d[maxed_at-1]
            lc = self.c[maxed_at-1]
            lb = self.b[maxed_at-1]

            sol_left = quad_solver(3*ld, 2*lc, lb)
            if sol_left.solvable:
                # Use solution with maximum (not minimum)
                # If the curve is mostly increasing, the maximal point is located at the smaller solution (i.e. sol1 for a>0).
                # If the curve is mostly decreasing, the maximal point is located at the larger solution (i.e. sol1 for a<0).
                chosen_sol = sol_left.sol1

                # Check if solution is within segment bounds
                # The spline coefficients are designed such that 'x' in 'y + b*x + c*x^2 + d*x^3' is
                # equal to 'x_t - x_l' where 'x_l' is the left limit of that spline segment and 'x_t'
                # is where you want to get an interpolated value. This is necessary in 'splinefun' to
                # ensure that you get 'y' (i.e. the original data point) when 'x=0'. For our purposes,
                # the actual MLE corresponds to 'x_t' and is equal to 'solution + x_0'.
                if chosen_sol > 0 and chosen_sol < x[maxed_at] - x[maxed_at-1]:
                    temp = ((ld*chosen_sol + lc)*chosen_sol + lb)*chosen_sol + y[maxed_at-1]
                    if temp > maxed:
                        maxed = temp
                        x_max = chosen_sol + x[maxed_at-1]

        # Check right segment for maximum
        if maxed_at < self.npts - 1:
            rd = self.d[maxed_at]
            rc = self.c[maxed_at]
            rb = self.b[maxed_at]

            sol_right = quad_solver(3*rd, 2*rc, rb)
            if sol_right.solvable:
                chosen_sol = sol_right.sol1

                if chosen_sol > 0 and chosen_sol < x[maxed_at+1] - x[maxed_at]:
                    temp = ((rd*chosen_sol + rc)*chosen_sol + rb)*chosen_sol + y[maxed_at]
                    if temp > maxed:
                        maxed = temp
                        x_max = chosen_sol + x[maxed_at]

        return x_max

    cdef void _fmm_spline(self, const double* x, const double* y):
        """
        Fit a cubic spline using Forsythe Malcolm Moler algorithm.

        In this case the end-conditions are determined by fitting cubic
        polynomials to the first and last 4 points and matching the third
        derivitives of the spline at the end-points to the third derivatives of
        these cubics at the end-points.

        This function is a direct port from splines.c in the R stats package.
        https://github.com/SurajGupta/r-source/blob/master/src/library/stats/src/splines.c

        Parameters
        ----------
        x : const double*
            Array of x-coordinates
        y : const double*
            Array of y-coordinates
        """
        cdef:
            Py_ssize_t nm1, i
            double t

        if self.npts < 2:
            raise ValueError("spline interpolation requires at least 2 points")

        if self.npts < 3:
            t = (y[1] - y[0])
            self.b[0] = t / (x[1] - x[0])
            self.b[1] = self.b[0]
            self.c[0] = self.c[1] = self.d[0] = self.d[1] = 0.0
            return

        nm1 = self.npts - 1

        # Set up tridiagonal system
        # b = diagonal, d = offdiagonal, c = right hand side
        self.d[0] = x[1] - x[0]
        self.c[1] = (y[1] - y[0]) / self.d[0]

        for i in range(1, self.npts-1):
            self.d[i] = x[i+1] - x[i]
            self.b[i] = 2.0 * (self.d[i-1] + self.d[i])
            self.c[i+1] = (y[i+1] - y[i]) / self.d[i]
            self.c[i] = self.c[i+1] - self.c[i]

        # End conditions
        # Third derivatives at x[0] and x[n-1] obtained from divided differences

        self.b[0] = -self.d[0]
        self.b[self.npts-1] = -self.d[nm1-1]
        self.c[0] = self.c[self.npts-1] = 0.0

        if self.npts > 3:
            self.c[0] = self.c[2] / (x[3] - x[1]) - self.c[1] / (x[2] - x[0])
            self.c[self.npts-1] = self.c[nm1-1] / (x[self.npts-1] - x[self.npts-3]) - self.c[self.npts-3] / (x[nm1-1] - x[self.npts-4])
            self.c[0] = self.c[0] * self.d[0] * self.d[0] / (x[3] - x[0])
            self.c[self.npts-1] = -self.c[self.npts-1] * self.d[nm1-1] * self.d[nm1-1] / (x[self.npts-1] - x[self.npts-4])

        # Gaussian elimination
        for i in range(1, self.npts):
            t = self.d[i-1] / self.b[i-1]
            self.b[i] = self.b[i] - t * self.d[i-1]
            self.c[i] = self.c[i] - t * self.c[i-1]

        # Backward substitution
        self.c[self.npts-1] = self.c[self.npts-1] / self.b[self.npts-1]
        for i in range(nm1-1, -1, -1):
            self.c[i] = (self.c[i] - self.d[i] * self.c[i+1]) / self.b[i]

        # c[i] is now the sigma[i-1] of the text
        # Compute polynomial coefficients
        self.b[self.npts-1] = (y[self.npts-1] - y[self.npts-2]) / self.d[self.npts-2] + self.d[self.npts-2] * (self.c[self.npts-2] + 2.0 * self.c[self.npts-1])
        for i in range(nm1):
            self.b[i] = (y[i+1] - y[i]) / self.d[i] - self.d[i] * (self.c[i+1] + 2.0 * self.c[i])
            self.d[i] = (self.c[i+1] - self.c[i]) / self.d[i]
            self.c[i] = 3.0 * self.c[i]

        self.c[self.npts-1] = 3.0 * self.c[self.npts-1]
        self.d[self.npts-1] = self.d[nm1-1]



@cython.wraparound(False)
@cython.boundscheck(False)
cpdef vector.vector[double] maximize_interpolant(vector.vector[double] spts, const double[:,:] likelihoods):
    """
    Find the maximum of interpolating splines for each row in the likelihood matrix.

    This is a Cython port of the maximize_interpolant function from edgeR.

    Parameters
    ----------
    spts : vector[double]
        Spline points (x-coordinates)
    likelihoods : const double[:,:]
        Matrix of likelihood values, where each row represents a different tag/gene
        and each column corresponds to a spline point

    Returns
    -------
    vector[double]
        Vector of x-coordinates where the maximum occurs for each tag/gene

    Raises
    ------
    RuntimeError
        If the number of columns in likelihood matrix doesn't match the number of spline points
    """
    cdef:
        Py_ssize_t num_pts = spts.size()
        Py_ssize_t num_tags = likelihoods.shape[0]
        Py_ssize_t tag, i
        interpolator maxinterpol
        vector.vector[double] current_ll
        vector.vector[double] all_spts
        vector.vector[double] output

    if num_pts != likelihoods.shape[1]:
        raise RuntimeError("number of columns in likelihood matrix should be equal to number of spline points")

    maxinterpol = interpolator(num_pts)
    current_ll.resize(num_pts)
    all_spts = spts  # Make a copy to guarantee contiguousness
    output.resize(num_tags)

    for tag in range(num_tags):
        # Copy current row to working array
        for i in range(num_pts):
            current_ll[i] = likelihoods[tag, i]

        output[tag] = maxinterpol.find_max(&all_spts[0], &current_ll[0])

    return output
