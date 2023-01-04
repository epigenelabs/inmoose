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

// This file is based on the file 'src/glm_levenberg.cpp' of the Bioconductor edgeR package (version 3.38.4).


#include "glm.h"

const double one_millionth=std::pow(10, -6.0);
const double supremely_low_value=std::pow(10, -13.0), ridiculously_low_value=std::pow(10, -100.0);

double glm_levenberg::nb_deviance (const double* y, const double* mu, const double* w, const double* phi) const {
    double tempdev=0;
    for (int i=0; i<nlibs; ++i) {
        tempdev+=w[i]*compute_unit_nb_deviance(y[i], mu[i], phi[i]);
    }
    return tempdev;
}

// Computing beta %*% design + offset.

const char trans='n';
const int incx=1, incy=1;
const double first_scaling=1, second_scaling=1;
void glm_levenberg::autofill(const double* beta, const double* offset, double* mu) {
    std::copy(offset, offset+nlibs, mu);
    f77_dgemv(&trans, &nlibs, &ncoefs, &first_scaling, design, &nlibs, beta, &incx, &second_scaling, mu, &incy);
	for (int lib=0; lib<nlibs; ++lib) {
		double& cur_mean=mu[lib];
		cur_mean=std::exp(cur_mean);
	}
	return;
}

// Constructors for the GLM object.

glm_levenberg::glm_levenberg(int nl, int nc, const double* d, long mi, double tol) : nlibs(nl), ncoefs(nc), maxit(mi), tolerance(tol),
        design(d), working_weights(nlibs), deriv(nlibs), xtwx(ncoefs*ncoefs), xtwx_copy(ncoefs*ncoefs), dl(ncoefs), dbeta(ncoefs),
        info(0), mu_new(nlibs), beta_new(ncoefs) {}

// Now, for the actual fit implementation.

const char uplo='U';
const int nrhs=1;

int glm_levenberg::fit(const double* y, const double* offset, const double* disp,
		const double* w, double* mu, double* beta) {
	// We expect 'beta' to be supplied. We then check the maximum value of the counts.
    double ymax=0;
    for (int lib=0; lib<nlibs; ++lib) {
		const double& count=y[lib];
		if (count>ymax) { ymax=count; }
 	}
    dev=0;
    iter=0;
	failed=false;

    // If we start off with all entries at zero, there's really no point continuing.
    if (ymax<low_value) {
        std::fill(beta, beta+ncoefs, std::numeric_limits<double>::quiet_NaN());
        std::fill(mu, mu+nlibs, 0);
        return 0;
    }

	// Otherwise, we compute 'mu' based on 'beta'. Returning if there are no coefficients!
	autofill(beta, offset, mu);
	dev=nb_deviance(y, mu, w, disp);
    if (ncoefs==0) {
        return 0;
    }

    // Iterating using reweighted least squares; setting up assorted temporary objects.
    double max_info=-1, lambda=0;
    while ((++iter) <= maxit) {

		/* Here we set up the matrix XtWX i.e. the Fisher information matrix. X is the design matrix and W is a diagonal matrix
 		 * with the working weights for each observation (i.e. library). The working weights are part of the first derivative of
 		 * the log-likelihood for a given coefficient, multiplied by any user-specified weights. When multiplied by two covariates
 		 * in the design matrix, you get the Fisher information (i.e. variance of the log-likelihood) for that pair. This takes
 		 * the role of the second derivative of the log-likelihood. The working weights are formed by taking the reciprocal of the
 		 * product of the variance (in terms of the mean) and the square of the derivative of the link function.
 		 *
 		 * We also set up the actual derivative of the log likelihoods in 'dl'. This is done by multiplying each covariate by the
 		 * difference between the mu and observation and dividing by the variance and derivative of the link function. This is
 		 * then summed across all observations for each coefficient. The aim is to solve (XtWX)(dbeta)=dl for 'dbeta'. As XtWX
 		 * is the second derivative, and dl is the first, you can see that we are effectively performing a multivariate
 		 * Newton-Raphson procedure with 'dbeta' as the step.
 		 */
        for (int lib=0; lib<nlibs; ++lib) {
            const double& cur_mu=mu[lib];
			const double denom=(1+cur_mu*disp[lib]);
            working_weights[lib]=cur_mu/denom*w[lib];
            deriv[lib]=(y[lib]-cur_mu)/denom*w[lib];
        }

        compute_xtwx(nlibs, ncoefs, design, working_weights.data(), xtwx.data());

        const double* dcopy=design;
        auto xtwxIt=xtwx.begin();
        for (int coef=0; coef<ncoefs; ++coef, dcopy+=nlibs, xtwxIt+=ncoefs) {
            dl[coef]=std::inner_product(deriv.begin(), deriv.end(), dcopy, 0.0);
            const double& cur_val=*(xtwxIt+coef);
            if (cur_val>max_info) { max_info=cur_val; }
        }
        if (iter==1) {
            lambda=max_info*one_millionth;
            if (lambda < supremely_low_value) { lambda=supremely_low_value; }
        }

        /* Levenberg/Marquardt damping reduces step size until the deviance increases or no
         * step can be found that increases the deviance. In short, increases in the deviance
         * are enforced to avoid problems with convergence.
         */
        int lev=0;
        bool low_dev=false;
        while (++lev) {
			do {
             	/* We need to set up copies as the decomposition routine overwrites the originals, and
 				 * we want the originals in case we don't like the latest step. For efficiency, we only
	 			 * refer to the upper triangular for the XtWX copy (as it should be symmetrical). We also add
	 			 * 'lambda' to the diagonals. This reduces the step size as the second derivative is increased.
        	     */
                auto xtwxIt=xtwx.begin(), xtwxcIt=xtwx_copy.begin();
         		for (int col=0; col<ncoefs; ++col, xtwxIt+=ncoefs, xtwxcIt+=ncoefs) {
                    std::copy(xtwxIt, xtwxIt+col+1, xtwxcIt);
                    *(xtwxcIt+col)+=lambda;
            	}

            	// Cholesky decomposition, and then use of the decomposition to solve for dbeta in (XtWX)dbeta = dl.
                f77_dpotrf(&uplo, &ncoefs, xtwx_copy.data(), &ncoefs, &info);
                if (info!=0) {
                    /* If it fails, it MUST mean that the matrix is singular due to numerical imprecision
                     * as all the diagonal entries of the XtWX matrix must be positive. This occurs because of
                     * fitted values being exactly zero; thus, the coefficients attempt to converge to negative
                     * infinity. This generally forces the step size to be larger (i.e. lambda lower) in order to
                     * get to infinity faster (which is impossible). Low lambda leads to numerical instability
                     * and effective singularity. To solve this, we actually increase lambda; this avoids code breakage
                     * to give the other coefficients a chance to converge. Failure of convergence for the zero-
                     * fitted values isn't a problem as the change in deviance from small --> smaller coefficients isn't
                     * that great when the true value is negative inifinity.
                     */
                    lambda*=10;
                	if (lambda <= 0) { lambda=ridiculously_low_value; } // Just to make sure it actually increases.
                } else {
                    break;
                }
            } while (1);

            std::copy(dl.begin(), dl.end(), dbeta.begin());
            f77_dpotrs(&uplo, &ncoefs, &nrhs, xtwx_copy.data(), &ncoefs, dbeta.data(), &ncoefs, &info FCONE);
            if (info!=0) {
                throw std::runtime_error("solution using the Cholesky decomposition failed");
            }

            // Updating beta and the means. 'dbeta' stores 'Y' from the solution of (X*VX)Y=dl, corresponding to a NR step.
            for (int coef=0; coef<ncoefs; ++coef) {
                beta_new[coef]=beta[coef]+dbeta[coef];
            }
            autofill(beta_new.data(), offset, mu_new.data());

            /* Checking if the deviance has decreased or if it's too small to care about. Either case is good
             * and means that we'll be using the updated fitted values and coefficients. Otherwise, if we have
             * to repeat the inner loop, then we want to do so from the original values (as we'll be scaling
             * lambda up so we want to retake the step from where we were before). This is why we don't modify the values
             * in-place until we're sure we want to take the step.
             */
            const double dev_new=nb_deviance(y, mu_new.data(), w, disp);
            if (dev_new/ymax < supremely_low_value) { low_dev=true; }
            if (dev_new <= dev || low_dev) {
                std::copy(beta_new.begin(), beta_new.end(), beta);
                std::copy(mu_new.begin(), mu_new.end(), mu);
                dev=dev_new;
                break;
            }

            // Increasing lambda, to increase damping. Again, we have to make sure it's not zero.
            lambda*=2;
            if (lambda <= 0) { lambda=ridiculously_low_value; }

            // Excessive damping; steps get so small that it's pointless to continue.
            if (lambda/max_info > 1/supremely_low_value) {
            	failed=1;
            	break;
            }
        }

        /* Terminating if we failed, if divergence from the exact solution is acceptably low
         * (cross-product of dbeta with the log-likelihood derivative) or if the actual deviance
         * of the fit is acceptably low.
         */
        if (failed) { break; }
		if (low_dev) { break; }
        const double divergence=std::inner_product(dl.begin(), dl.end(), dbeta.begin(), 0.0);
        if (divergence < tolerance) { break; }

        /* If we quit the inner levenberg loop immediately and survived all the break conditions above, that means that deviance is decreasing
 		 * substantially. Thus, we need larger steps to get there faster. To do so, we decrease the damping factor. Note that this only applies
 		 * if we didn't decrease the damping factor in the inner levenberg loop, as that would indicate that we need to slow down.
         */
        if (lev==1) { lambda/=10; }
    }
	return 0;
}

/* Finally, assorted getters. */

const double& glm_levenberg::get_deviance() const {return dev; }

const long& glm_levenberg::get_iterations() const { return iter; }

const bool& glm_levenberg::is_failure() const { return failed; }

