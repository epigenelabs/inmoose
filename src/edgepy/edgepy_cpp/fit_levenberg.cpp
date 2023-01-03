#include "glm.h"
#include "numpy/ndarraytypes.h"
#include "objects.h"

PyObject* fit_levenberg (PyArrayObject* y, PyArrayObject* offset, PyArrayObject* disp, PyArrayObject* weights, PyArrayObject* design, PyArrayObject* beta, double tol, long maxit) {

    any_numeric_matrix counts(y);
    const size_t num_tags=counts.get_nrow();
    const size_t num_libs=counts.get_ncol();

    // Getting and checking the dimensions of the arguments.
    NumericMatrix X=check_design_matrix(design, num_libs);
    const size_t num_coefs=X.get_ncol();

    NumericMatrix Beta(beta);
    if (Beta.get_nrow()!=num_tags || Beta.get_ncol()!=num_coefs) {
        throw std::runtime_error("dimensions of beta starting values are not consistent with other dimensions");
    }

    // Initializing pointers to the assorted features.
    compressed_matrix allo=check_CM_dims(offset, num_tags, num_libs, "offset", "count");
    compressed_matrix alld=check_CM_dims(disp, num_tags, num_libs, "dispersion", "count");
    compressed_matrix allw=check_CM_dims(weights, num_tags, num_libs, "weight", "count");

    // Setting up scalars.
    long max_it=maxit;
    double tolerance=tol;

    // Initializing output objects.
    NumericMatrix out_beta(num_tags, num_coefs);
    NumericMatrix out_fitted(num_tags, num_libs);
    NumericVector out_dev(num_tags);
    IntegerVector out_iter(num_tags);
    LogicalVector out_conv(num_tags);

    std::vector<double> current(num_libs), tmp_beta(num_coefs), tmp_fitted(num_libs);
	  glm_levenberg glbg(num_libs, num_coefs, X.begin(), max_it, tolerance);

    for (size_t tag=0; tag<num_tags; ++tag) {
        counts.fill_row(tag, current.data());
        auto beta_row=Beta.row(tag);
        std::copy(beta_row.begin(), beta_row.end(), tmp_beta.begin());

        if (glbg.fit(current.data(), allo.get_row(tag), alld.get_row(tag), allw.get_row(tag), tmp_fitted.data(), tmp_beta.data())) {
            std::stringstream errout;
            errout<< "solution using Cholesky decomposition failed for tag " << tag+1;
            throw std::runtime_error(errout.str());
        }

        std::copy(tmp_fitted.begin(), tmp_fitted.end(), out_fitted.row(tag).begin());
        std::copy(tmp_beta.begin(), tmp_beta.end(), out_beta.row(tag).begin());

		out_dev[tag]=glbg.get_deviance();
		out_iter[tag]=glbg.get_iterations();
		out_conv[tag]=glbg.is_failure();
    }

    return make_levenberg_result(
        out_beta.to_ndarray(),
        out_fitted.to_ndarray(),
        out_dev,
        out_iter,
        out_conv);
}
