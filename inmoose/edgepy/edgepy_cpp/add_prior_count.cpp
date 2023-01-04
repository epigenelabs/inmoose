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

// This file is based on the file 'src/R_add_prior_count.cpp' of the Bioconductor edgeR package (version 3.38.4).


#include "numpy/ndarraytypes.h"
#include "utils.h"
#include "add_prior.h"
#include <limits>

/**** Adding a prior count to each observation. *******/

PyObject* add_prior_count(PyArrayObject* y, PyArrayObject* offset, PyArrayObject* prior) {
    any_numeric_matrix input(y);
    const size_t num_tags=input.get_nrow();
    const size_t num_libs=input.get_ncol();

    NumericMatrix outmat(num_tags, num_libs);
    if (input.is_data_integer()) {
        const auto& tmp=input.get_raw_int();
        std::copy(tmp.begin(), tmp.end(), outmat.begin());
    } else {
        const auto& tmp=input.get_raw_dbl();
        std::copy(tmp.begin(), tmp.end(), outmat.begin());
    }

    add_prior AP(prior, offset, true, true);
    check_AP_dims(AP, num_tags, num_libs, "count");

    // Computing the adjusted library sizes, either as a vector or as a matrix.
    const bool same_prior=AP.same_across_rows();
    PyArrayObject* output;

    if (num_tags == 0) {
        if (same_prior) {
             output=vector2ndarray(NumericVector(num_libs, std::numeric_limits<double>::quiet_NaN()));
        } else {
             output=NumericMatrix(num_tags, num_libs).to_ndarray();
        }
        return PyTuple_Pack(2, outmat.to_ndarray(), output);
    }

    NumericMatrix offset_mat(num_tags, num_libs);
    // Adding the prior values to the existing counts.
    for (size_t tag=0; tag<num_tags; ++tag) {
        AP.compute(tag);
        const double* pptr=AP.get_priors();

        auto current=outmat.row(tag);
        for (auto& curval : current) {
            curval += *pptr;
            ++pptr;
        }

        if (!same_prior) {
            const double* optr=AP.get_offsets();
            auto current=offset_mat.row(tag);

            for (auto& curval : current) {
               curval = *optr;
               ++optr;
            }
        }
    }

    if (same_prior) {
        AP.compute(0);
        const double* optr=AP.get_offsets();
        output=vector2ndarray(NumericVector(optr, optr+num_libs));
    } else {
        output=offset_mat.to_ndarray();
    }

    return PyTuple_Pack(2, outmat.to_ndarray(), output);
}

