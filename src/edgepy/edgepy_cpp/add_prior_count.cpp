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

