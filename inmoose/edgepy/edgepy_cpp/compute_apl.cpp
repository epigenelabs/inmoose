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

// This file is based on the file 'src/R_compute_apl.cpp' of the Bioconductor edgeR package (version 3.38.4).


#include "glm.h"
#include "objects.h"

NumericVector compute_apl(PyArrayObject* y, PyArrayObject* means, PyArrayObject* disps, PyArrayObject* weights, bool adjust, PyArrayObject* design) {
    any_numeric_matrix counts(y);
    const size_t num_tags=counts.get_nrow();
    const size_t num_libs=counts.get_ncol();

    // Setting up the means.
    NumericMatrix Means(means);
    if (Means.get_nrow()!=num_tags || Means.get_ncol()!=num_libs) {
        throw std::runtime_error("fitted value and count matrices must have the same dimensions");
    }

    // Setting up the dispersions and weights.
    compressed_matrix alld=check_CM_dims(disps, num_tags, num_libs, "dispersion", "count");
    compressed_matrix allw=check_CM_dims(weights, num_tags, num_libs, "weight", "count");

    // Determining whether we want to do the adjustment.
    bool do_adjust=adjust;

    // Setting up the CR adjustment object.
    NumericMatrix X=check_design_matrix(design, num_libs);
    const size_t num_coefs=X.get_ncol();
    adj_coxreid acr(num_libs, num_coefs, X.begin());

    // Generating output values.
    NumericVector output(num_tags);
    std::vector<double> working_weights(num_libs), current(num_libs);
    for (size_t tag=0; tag<num_tags; ++tag) {

        double& sum_loglik=output[tag];
        counts.fill_row(tag, current.data());
        auto curmeans=Means.row(tag);
        auto dptr=alld.row(tag);
        auto wptr=allw.row(tag);

        /* First computing the log-likelihood. */
        auto cmIt=curmeans.begin();
        for (size_t lib=0; lib<num_libs; ++lib, ++cmIt) {
            if ((*cmIt)==0) {
                if (do_adjust) {
                    working_weights[lib] = 0;
                }
                continue; // Mean should only be zero if count is zero, where the log-likelihood would then be 0.
            }

            // Each y is assumed to be the average of 'weights' counts, so we convert
            // from averages to the "original sums" in order to compute NB probabilities.
            const double& curw = wptr[lib];
            const double curmu = (*cmIt) * curw;
            const double cury = current[lib] * curw;
            const double curd = dptr[lib] / curw;

            double loglik=0;
            if (curd > 0) {
                // same as loglik <- rowSums(weights*dnbinom(y,size=1/dispersion,mu=mu,log = TRUE))
                const double r=1/curd;
                const double logmur=std::log(curmu+r);
                loglik = cury*std::log(curmu) - cury*logmur + r*std::log(r) - r*logmur + lgamma(cury+r) - lgamma(cury+1) - lgamma(r);
            } else {
                // same as loglik <- rowSums(weights*dpois(y,lambda=mu,log = TRUE))
                loglik = cury*std::log(curmu) - curmu - lgamma(cury+1);
            }
            sum_loglik += loglik;

            // Adding the Jacobian, to account for the fact that we actually want the log-likelihood
            // of the _scaled_ NB distribution (after dividing the original sum by the weight).
            sum_loglik += std::log(curw);

            if (do_adjust) {
                /* Computing 'W', the matrix of negative binomial working weights.
                 * The class computes 'XtWX' and performs an LDL decomposition
                 * to compute the Cox-Reid adjustment factor.
                 */
                working_weights[lib] = curmu/(1 + curd * curmu);
            }
        }

        if (do_adjust) {
            double adj=0;
            if (num_coefs==1) {
                adj=std::accumulate(working_weights.begin(), working_weights.end(), 0.0);
                adj=std::log(std::abs(adj))/2;
            } else {
                std::pair<double, bool> x=acr.compute(working_weights.data());
                if (!x.second) {
                    std::stringstream errout;
                    errout << "LDL factorization failed for tag " << tag+1;
                    throw std::runtime_error(errout.str());
                }
                adj=x.first;
            }
            sum_loglik-=adj;
        }
    }

    return output;
}

