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

// This file is based on the file 'src/add_prior.h' of the Bioconductor edgeR package (version 3.38.4).


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
