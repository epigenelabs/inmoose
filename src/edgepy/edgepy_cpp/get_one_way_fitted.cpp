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

// This file is based on the file 'src/R_get_one_way_fitted.cpp' of the Bioconductor edgeR package (version 3.38.4).


#include "glm.h"
#include "numpy/ndarraytypes.h"
#include "objects.h"

/*** Function to compute the fitted values without a lot of temporary matrices. ***/

PyArrayObject* get_one_way_fitted (PyArrayObject* beta, PyArrayObject* offset, std::vector<int> groups) {
    const NumericMatrix Beta(beta);
    const size_t num_tags=Beta.get_nrow();
    const size_t num_groups=Beta.get_ncol();

    const size_t num_libs=groups.size();
    if (*std::min_element(groups.begin(), groups.end()) < 0) {
        throw std::runtime_error("smallest value of group vector should be non-negative");
    }
    if ((size_t)*std::max_element(groups.begin(), groups.end()) >= num_groups) {
        throw std::runtime_error("largest value of group vector should be less than the number of groups");
    }

    compressed_matrix allo=check_CM_dims(offset, num_tags, num_libs, "offset", "count");

    NumericMatrix output(num_tags, num_libs);
    // output[i,j] = exp(allo[i,j] + Beta[i,groups[j]])
    auto outIt = output.begin();
    auto alloIt = allo.begin();
    for (auto gIt = groups.begin(); gIt != groups.end(); ++gIt) {
        auto curbeta = Beta.col(*gIt);
        auto betaIt = curbeta.begin();
        for (auto betaIt = curbeta.begin(); betaIt != curbeta.end(); ++betaIt, ++outIt, ++alloIt) {
           (*outIt) = std::exp(*alloIt + *betaIt);
        }
    }

    return output.to_ndarray();
}

