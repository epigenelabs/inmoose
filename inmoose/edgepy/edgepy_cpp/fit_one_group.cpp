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

// This file is based on the file 'src/R_fit_one_group.cpp' of the Bioconductor edgeR package (version 3.38.4).


#include "glm.h"
#include "numpy/ndarraytypes.h"
#include "objects.h"

template <typename T>
bool is_array_equal_to(const T* x, const size_t n, const bool rep, const T& v) {
    if (rep) {
        return (n>0 && x[0]==v);
    } else {
        for (size_t i=0; i<n; ++i) {
            if (x[i]!=v) { return false; }
        }
        return true;
    }
}

std::pair<NumericVector, LogicalVector>
fit_one_group (PyArrayObject* y, PyArrayObject* offsets, PyArrayObject* disp, PyArrayObject* weights, long max_iterations, double tolerance, NumericVector beta) {
    any_numeric_matrix counts(y);
    const size_t num_tags=counts.get_nrow();
    const size_t num_libs=counts.get_ncol();
    std::vector<double> current(num_libs);

    // Setting up assorted input matrices.
    compressed_matrix allo=check_CM_dims(offsets, num_tags, num_libs, "offset", "count");
    compressed_matrix alld=check_CM_dims(disp, num_tags, num_libs, "dispersion", "count");
    compressed_matrix allw=check_CM_dims(weights, num_tags, num_libs, "weight", "count");

    // Setting up the beta object.
    NumericVector Beta(beta);
    if (Beta.size()!=num_tags) {
        throw std::runtime_error("length of beta vector should be equal to number of genes");
    }

    // Setting up scalars.
    long maxit=max_iterations;
    double tol=tolerance;

    // Setting up beta for output.
    NumericVector out_beta(num_tags);
    LogicalVector out_conv(num_tags);

    // Preparing for possible Poisson sums.
    bool disp_is_zero=true, weight_is_one=true;
    double sum_lib=0;
    if (allo.is_row_repeated() && num_tags) {
        auto optr=allo.row(0);
        for (size_t lib=0; lib<num_libs; ++lib) {
            sum_lib+=std::exp(optr[lib]);
        }
     }
    if (alld.is_row_repeated() && num_tags) {
        const double* dptr=alld.get_row(0);
        disp_is_zero=is_array_equal_to<double>(dptr, num_libs, alld.is_col_repeated(), 0);
    }
    if (allw.is_row_repeated() && num_tags) {
        const double* wptr=allw.get_row(0);
        weight_is_one=is_array_equal_to<double>(wptr, num_libs, allw.is_col_repeated(), 1);
    }

    // Iterating through tags and fitting.
	for (size_t tag=0; tag<num_tags; ++tag) {
        counts.fill_row(tag, current.data());
        const double* optr=allo.get_row(tag);
        const double* wptr=allw.get_row(tag);
        const double* dptr=alld.get_row(tag);

        // Checking for the Poisson special case with all-unity weights and all-zero dispersions.
        if (!alld.is_row_repeated()) {
            disp_is_zero=is_array_equal_to<double>(dptr, num_libs, alld.is_col_repeated(), 0);
        }
        if (!allw.is_row_repeated()) {
            weight_is_one=is_array_equal_to<double>(wptr, num_libs, allw.is_col_repeated(), 1);
        }

        if (disp_is_zero && weight_is_one) {
            if (!allo.is_row_repeated()) {
                // Only recalculate sum of library sizes if it has changed.
                sum_lib=0;
                for (size_t lib=0; lib<num_libs; ++lib) { sum_lib+=std::exp(optr[lib]); }
            }

            double sum_counts=std::accumulate(current.begin(), current.end(), 0.0);
            if (sum_counts==0) {
                out_beta[tag]=neg_inf;
            } else {
                out_beta[tag]=std::log(sum_counts/sum_lib);
            }
            out_conv[tag]=true;
        } else {
            // Otherwise going through NR iterations.
            std::pair<double, bool> out=glm_one_group(num_libs, current.data(), optr, dptr, wptr, maxit, tol, Beta[tag]);
            out_beta[tag]=out.first;
            out_conv[tag]=out.second;
        }
	}

	// Returning everything as a pair
  return std::make_pair(out_beta, out_conv);
}
