#ifndef ADD_PRIOR_H
#define ADD_PRIOR_H
#include "objects.h"
#include "utils.h"

class add_prior{
public:
    add_prior(PyArrayObject*, PyArrayObject*, bool, bool);
    void compute(size_t);
    const double* get_priors() const;
    const double* get_offsets() const;

    size_t get_nrow() const;
    size_t get_ncol() const;
    const bool same_across_rows() const;
private:
    compressed_matrix allp, allo;
    const bool logged_in, logged_out;
    size_t nrow, ncol;

    std::vector<double> adj_prior, adj_libs;
    bool filled;
};

void check_AP_dims(const add_prior&, size_t, size_t, const char*);

#endif
