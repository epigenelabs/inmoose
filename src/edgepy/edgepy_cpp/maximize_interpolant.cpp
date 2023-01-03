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

// This file is based on the file 'src/R_maximize_interpolant.cpp' of the Bioconductor edgeR package (version 3.38.4).


#include "utils.h"
#include "interpolator.h"

std::vector<double> maximize_interpolant(std::vector<double> spts, PyArrayObject* likelihoods) {
    NumericMatrix ll(likelihoods);
    const size_t num_pts=spts.size();
    if (num_pts!=ll.get_ncol()) {
        throw std::runtime_error("number of columns in likelihood matrix should be equal to number of spline points");
    }
    const size_t num_tags=ll.get_nrow();

    interpolator maxinterpol(num_pts);
    std::vector<double> current_ll(num_pts);
    std::vector<double> all_spts(spts.begin(), spts.end()); // making a copy to guarantee contiguousness.

    NumericVector output(num_tags);
    for (size_t tag=0; tag<num_tags; ++tag) {
        auto curll=ll.row(tag);
        std::copy(curll.begin(), curll.end(), current_ll.begin());
        output[tag]=maxinterpol.find_max(all_spts.data(), current_ll.data());
    }

    return(output);
}
