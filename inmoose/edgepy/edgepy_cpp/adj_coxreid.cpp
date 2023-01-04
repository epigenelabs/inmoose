// Copyright (C) 2008-2022 Yunshun Chen, Aaron TL Lun, Davis J McCarthy, Matthew E Ritchie, Belinda Phipson, Yifang Hu, Xiaobei Zhou, Mark D Robinson, Gordon K Smyth
// Copyright (C) 2022-2023 Maximilien Colange
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

// This file is based on the file 'src/adj_coxreid.cpp' of the Bioconductor edgeR package (version 3.38.4).


#include "glm.h"

const char uplo='U';

adj_coxreid::adj_coxreid (int nl, int nc, const double* d) : ncoefs(nc), nlibs(nl),
        design(d), xtwx(ncoefs*ncoefs), pivots(ncoefs),
        info(0), lwork(-1) {

    /* We also want to identify the optimal size of the 'work' array
     * using the ability of the dystrf function to call ILAENV. We then
     * reallocate the work pointer to this value.
     */
	double temp_work;
    f77_dsytrf(&uplo, &ncoefs, xtwx.data(), &ncoefs, pivots.data(), &temp_work, &lwork, &info FCONE);
	if (info) {
		throw std::runtime_error("failed to identify optimal size of workspace through ILAENV");
	}
    lwork=int(temp_work+0.5);
    if (lwork < 1) { lwork = 1; }
    work.resize(lwork);
	return;
}

std::pair<double, bool> adj_coxreid::compute(const double* wptr) {
    // Setting working weight_matrix to 'A=Xt %*% diag(W) %*% X' with column-major storage for the lower-triangular.
    // Then doing and LDL* decomposition, see details below.
    compute_xtwx(nlibs, ncoefs, design, wptr, xtwx.data());
    f77_dsytrf(&uplo, &ncoefs, xtwx.data(), &ncoefs, pivots.data(), work.data(), &lwork, &info FCONE);
    if (info<0) { return std::make_pair(0, false); }

    // Log-determinant as sum of the log-diagonals, then halving (see below).
    auto wmIt=xtwx.begin();
    double sum_log_diagonals=0;
    for (int i=0; i<ncoefs; ++i, wmIt+=ncoefs) {
        const double& cur_val=*(wmIt+i);
		if (cur_val < low_value || !std::isfinite(cur_val))  {
			sum_log_diagonals += log_low_value;
		} else {
			sum_log_diagonals += std::log(cur_val);
		}
    }
	return std::make_pair(sum_log_diagonals*0.5, true);
}

/* EXPLANATION:

   XtWX represents the expected Fisher information. The overall strategy is to compute the
   log-determinant of this matrix, to compute the adjustment factor for the likelihood (in
   order to account for uncertainty in the nuisance parameters i.e. the fitted values).

   We want to apply the Cholesky decomposition to the XtWX matrix. However, to be safe,
   we call the routine to do a symmetric indefinite factorisation i.e. A = LDLt. This
   guarantees factorization for singular matrices when the actual Cholesky decomposition
   would fail because it would start whining about non-positive eigenvectors.

   We then try to compute the determinant of XtWX. Here we use two facts:

   - For triangular matrices, the determinant is the product of the diagonals.
   - det(LDL*)=det(L)*det(D)*det(L*)
   - All diagonal elements of 'L' are unity.

   Thus, all we need to do is to we sum over all log'd diagonal elements in 'D', which -
   happily enough - are stored as the diagonal elements of 'xtwx'. (And then
   divide by two, because that's just how the Cox-Reid adjustment works.)

   'info > 0' indicates that one of the diagonals is zero. We handle this by replacing the
   it with an appropriately small non-zero value, if the diagonal element is zero or NA. This
   is valid because the zero elements correspond to all-zero columns in "WX", which in turn
   only arise when there are fitted values of zero, which will be constant at all dispersions.
   Thus, any replacement value will eventually cancel out during interpolation to obtain the CRAPLE.

   Note that the LAPACK routine will also do some pivoting, essentially solving PAP* = LDL* for
   some permutation matrix P. This shouldn't affect anything; the determinant of the permutation
   is either 1 or -1, but this cancels out, so det(A) = det(PAP*).

   Further note that the routine can theoretically give block diagonals, but this should
   not occur for positive (semi)definite matrices, which is what XtWX should always be.
*/

// Computes upper-triangular matrix.
void compute_xtwx(int nlibs, int ncoefs, const double* X, const double* W, double* out) {
    const double* xptr1=X;
    for (int coef1=0; coef1<ncoefs; ++coef1, xptr1+=nlibs) {
        const double* xptr2=X;
        for (int coef2=0; coef2<=coef1; ++coef2, xptr2+=nlibs) {

            double& cur_entry=(out[coef2]=0);
            for (int lib=0; lib<nlibs; ++lib) {
                cur_entry+=xptr1[lib]*xptr2[lib]*W[lib];
            }
        }
        out+=ncoefs;
    }
    return;
}
