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

// This file is based on the file 'src/R_ave_log_cpm.cpp' of the Bioconductor edgeR package (version 3.38.4).


#include "glm.h"
#include "add_prior.h"
#include "numpy/ndarraytypes.h"
#include "objects.h"
#include <limits>

std::vector<double> ave_log_cpm(PyArrayObject* y, PyArrayObject* offset, PyArrayObject* prior, PyArrayObject* disp, PyArrayObject* weights, long max_iterations, double tolerance) {
    any_numeric_matrix counts(y);
    const size_t num_tags=counts.get_nrow();
    const size_t num_libs=counts.get_ncol();
    std::vector<double> current(num_libs);

    add_prior AP(prior, offset, true, true);
    check_AP_dims(AP, num_tags, num_libs, "count");
    compressed_matrix alld=check_CM_dims(disp, num_tags, num_libs, "dispersion", "count");
    compressed_matrix allw=check_CM_dims(weights, num_tags, num_libs, "weight", "count");

    // GLM fitting specifications
    long maxit=max_iterations;
    double tol=tolerance;

    // Returning average log-cpm
    NumericVector output(num_tags);
    for (size_t tag=0; tag<num_tags; ++tag) {
        counts.fill_row(tag, current.data());

        // Adding the current set of priors.
        AP.compute(tag);
        const double* offptr=AP.get_offsets();
        const double* pptr=AP.get_priors();
        for (size_t lib=0; lib<num_libs; ++lib) {
            current[lib]+=pptr[lib];
        }

        // Fitting a one-way layout.
        const double* dptr=alld.get_row(tag);
        const double* wptr=allw.get_row(tag);
        auto fit=glm_one_group(num_libs, current.data(), offptr, dptr, wptr, maxit, tol, std::numeric_limits<double>::quiet_NaN());
        output[tag]=(fit.first + LNmillion)/LNtwo;
    }

    return output;
}
