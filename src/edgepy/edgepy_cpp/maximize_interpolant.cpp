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
