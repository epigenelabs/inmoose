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

from libcpp cimport bool, pair, vector
from numpy cimport ndarray

# cf. file src/utils.h from edgeR source repo

cdef public ndarray[double, ndim=1] vector2ndarray "vector2ndarray"(const vector.vector[double]& data)

cdef extern from "objects.h":
    cpdef bool is_integer_array(ndarray arr) except +

cdef extern from "add_prior_count.h":
    # y dtype is either int or double
    cpdef object cxx_add_prior_count "add_prior_count"(
            ndarray y,
            ndarray[double] offset,
            ndarray[double] prior,
            ) except +

cdef extern from "compute_nbdev.cpp":
    # y dtype is either int or double
    cpdef vector.vector[double] cxx_compute_nbdev_sum "compute_nbdev_sum"(
            ndarray y,
            ndarray[double] mu,
            ndarray[double] phi,
            ndarray[double] weights) except +
    # y dtype is either int or double
    cpdef ndarray[double] cxx_compute_nbdev_nosum "compute_nbdev_nosum"(
            ndarray y,
            ndarray[double] mu,
            ndarray[double] phi,
            ndarray[double] weights) except +

cdef extern from "maximize_interpolant.cpp":
    cpdef vector.vector[double] cxx_maximize_interpolant "maximize_interpolant"(vector.vector[double] spts, ndarray[double] likelihoods) except +

cdef extern from "get_one_way_fitted.cpp":
    cpdef ndarray[double] cxx_get_one_way_fitted "get_one_way_fitted"(
            ndarray[double] beta,
            ndarray[double] offset,
            vector.vector[int] groups) except +

cdef extern from "fit_one_group.cpp":
    # y dtype is either int or double
    cpdef pair.pair[vector.vector[double], vector.vector[char]] cxx_fit_one_group "fit_one_group"(
            ndarray y,
            ndarray[double] offsets,
            ndarray[double] disp,
            ndarray[double] weights,
            long max_iterations,
            double tolerance,
            vector.vector[double] beta) except +

cdef extern from "ave_log_cpm.cpp":
    # y dtype is either int or double
    cpdef vector.vector[double] cxx_ave_log_cpm "ave_log_cpm"(
            ndarray y,
            ndarray[double] offset,
            ndarray[double] prior,
            ndarray[double] disp,
            ndarray[double] weights,
            long max_iterations,
            double tolerance) except +

# Helper function to return from C++ fit_levenberg function
cdef public object make_levenberg_result "make_levenberg_result"(
        ndarray[double, ndim=2] coefficients,
        ndarray[double, ndim=2] fitted_values,
        vector.vector[double] deviance,
        vector.vector[long] iter,
        vector.vector[char] failed)

cdef extern from "fit_levenberg.h":
    # y dtype is either int or double
    cpdef object cxx_fit_levenberg "fit_levenberg"(
            ndarray y,
            ndarray[double] offset,
            ndarray[double] disp,
            ndarray[double] weights,
            ndarray[double] design,
            ndarray[double] beta,
            double tol,
            long maxit) except +

cdef extern from "compute_apl.cpp":
    # y dtype is either int or double
    cpdef vector.vector[double] cxx_compute_apl "compute_apl"(
            ndarray y,
            ndarray[double] means,
            ndarray[double] disps,
            ndarray[double] weights,
            bool adjust,
            ndarray[double] design) except +

cdef extern from "initialize_levenberg.cpp":
    # y dtype is either int or double
    cpdef ndarray cxx_get_levenberg_start "get_levenberg_start"(
            ndarray y,
            ndarray[double] offset,
            ndarray[double] disp,
            ndarray[double] weights,
            ndarray[double] design,
            bool use_null) except +

cdef extern from "initialize_levenberg.h":
    cdef cppclass QRdecomposition:
        int NR
        int NC
        double* X
        vector.vector[double] Xcopy
        vector.vector[double] tau
        vector.vector[double] effects
        vector.vector[double] weights
        vector.vector[double] work_geqrf
        vector.vector[double] work_ormqr
        int lwork_geqrf
        int lwork_ormqr
        int info

# Indicate the C name to workaround a Cython bug
# see https://github.com/cython/cython/issues/2940
cdef public void build_QRdecomposition "build_QRdecomposition"(QRdecomposition* self)
cdef public void decompose_QRdecomposition "decompose_QRdecomposition"(QRdecomposition* self)
cdef public void solve_QRdecomposition "solve_QRdecomposition"(QRdecomposition* self)

# Re-expose in C BLAS/LAPACK functions exposed in Cython by scipy
cdef public void f77_dgemv "f77_dgemv"(const char *trans, const int *m, const int *n, const double *alpha, const double *a, const int *lda, const double *x, const int *incx, const double *beta, double *y, const int *incy) nogil
cdef public void f77_dpotrf "f77_dpotrf"(const char *uplo, const int *n, double *a, const int *lda, int *info) nogil
cdef public void f77_dpotrs "f77_dpotrs"(const char *uplo, const int *n, const int *nrhs, const double *a, const int *lda, double *b, const int *ldb, int *info) nogil
cdef public void f77_dsytrf "f77_dsytrf"(const char *uplo, const int *n, double *a, const int *lda, int *ipiv, double *work, const int *lwork, int *info) nogil

