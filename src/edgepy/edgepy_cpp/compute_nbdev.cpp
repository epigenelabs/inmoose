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

// This file is based on the file 'src/R_compute_nbdev.cpp' of the Bioconductor edgeR package (version 3.38.4).


#include "utils.h"
#include "glm.h"
#include "objects.h"

template<class T>
std::vector<double>
_compute_nbdev_sum(const Matrix<T>& counts, PyArrayObject* mu, PyArrayObject* phi, PyArrayObject* weights) {
    const size_t num_tags=counts.get_nrow();
    const size_t num_libs=counts.get_ncol();
    std::vector<double> current(num_libs);

    // Setting up means.
    NumericMatrix fitted(mu);
    if (fitted.get_nrow()!=num_tags || fitted.get_ncol()!=num_libs) {
        throw std::runtime_error("dimensions of count and fitted value matrices are not equal");
    }

    // Setting up dispersions.
    compressed_matrix alld=check_CM_dims(phi, num_tags, num_libs, "dispersion", "count");

    // Setting up weights.
    compressed_matrix allw(weights);

    NumericVector output(num_tags);
    // output[i] = sum_j allw[i,j] * compute_unit_nb_deviance(counts[i,j], fitted[i,j], alld[i,j])
    auto wIt = allw.begin();
    auto countsIt = counts.begin();
    auto cmIt = fitted.begin();
    auto dIt = alld.begin();
    for (size_t lib=0; lib<num_libs; ++lib) {
        for (auto sumdevIt = output.begin(); sumdevIt != output.end(); ++sumdevIt, ++wIt, ++countsIt, ++cmIt, ++dIt) {
            (*sumdevIt) += compute_unit_nb_deviance(*countsIt, *cmIt, *dIt) * (*wIt);
        }
    }

    return output;
}

std::vector<double> compute_nbdev_sum(PyArrayObject* y, PyArrayObject* mu, PyArrayObject* phi, PyArrayObject* weights) {
    any_numeric_matrix counts(y);
    if (counts.is_data_integer()) {
      return _compute_nbdev_sum(counts.get_raw_int(), mu, phi, weights);
    } else {
      return _compute_nbdev_sum(counts.get_raw_dbl(), mu, phi, weights);
    }
}

PyArrayObject* compute_nbdev_nosum(PyArrayObject* y, PyArrayObject* mu, PyArrayObject* phi, PyArrayObject* weights) {
    any_numeric_matrix counts(y);
    const size_t num_tags=counts.get_nrow();
    const size_t num_libs=counts.get_ncol();
    std::vector<double> current(num_libs);

    // Setting up means.
    NumericMatrix fitted(mu);
    if (fitted.get_nrow()!=num_tags || fitted.get_ncol()!=num_libs) {
        throw std::runtime_error("dimensions of count and fitted value matrices are not equal");
    }

    // Setting up dispersions.
    compressed_matrix alld=check_CM_dims(phi, num_tags, num_libs, "dispersion", "count");

    NumericMatrix output(num_tags, num_libs);
    for (size_t tag=0; tag<num_tags; ++tag) {
        counts.fill_row(tag, current.data());
        auto dptr=alld.row(tag);

        auto curmeans=fitted.row(tag);
        auto cmIt=curmeans.begin();
        auto outvals=output.row(tag);
        auto ovIt=outvals.begin();

        for (size_t lib=0; lib<num_libs; ++lib, ++ovIt, ++cmIt) {
            (*ovIt) = compute_unit_nb_deviance(current[lib], *cmIt, dptr[lib]);
        }
   }

    return output.to_ndarray();
}
