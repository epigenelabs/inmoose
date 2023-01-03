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

// This file is based on the file 'src/glm_one_group.cpp' of the Bioconductor edgeR package (version 3.38.4).


#include "glm.h"
#include "utils.h"

std::pair<double,bool> glm_one_group(size_t nlibs, const double* counts, const double* offset,
        const double* disp, const double* weights, long maxit, double tolerance, double cur_beta) {
    /* Setting up initial values for beta as the log of the mean of the ratio of counts to offsets.
 	 * This is the exact solution for the gamma distribution (which is the limit of the NB as
 	 * the dispersion goes to infinity. However, if cur_beta is not NA, then we assume it's good.
 	 */
	bool nonzero=false;
	if (isnan(cur_beta)) {
		cur_beta=0;
 	   	double totweight=0;
		for (size_t j=0; j<nlibs; ++j) {
			const double& cur_val=counts[j];
			if (cur_val>low_value) {
				cur_beta+=cur_val/std::exp(offset[j]) * weights[j];
				nonzero=true;
			}
			totweight+=weights[j];
		}
		cur_beta=std::log(cur_beta/totweight);
	} else {
		for (size_t j=0; j<nlibs; ++j) {
			if (counts[j] > low_value) {
                nonzero=true;
                break;
            }
		}
	}

	// Skipping to a result for all-zero rows.
	if (!nonzero) {
        return std::make_pair(neg_inf, true);
    }

	// Newton-Raphson iterations to converge to mean.
    bool has_converged=false;
	for (long i=0; i<maxit; ++i) {
		double dl=0;
 	    double info=0;
		for (size_t j=0; j<nlibs; ++j) {
			const double mu=std::exp(cur_beta+offset[j]), denominator=1+mu*disp[j];
			dl+=(counts[j]-mu)/denominator * weights[j];
			info+=mu/denominator * weights[j];
		}
		const double step=dl/info;
		cur_beta+=step;
		if (std::abs(step)<tolerance) {
			has_converged=true;
			break;
		}
	}

	return std::make_pair(cur_beta, has_converged);
}


