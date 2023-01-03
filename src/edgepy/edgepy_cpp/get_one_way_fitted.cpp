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

