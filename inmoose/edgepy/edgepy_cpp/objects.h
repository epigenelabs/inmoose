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

// This file is based on the file 'src/objects.h' of the Bioconductor edgeR package (version 3.38.4).


#include "utils.h"
#include "matrix.h"
#ifndef OBJECTS_H
#define OBJECTS_H


class any_numeric_matrix {
public:
    any_numeric_matrix(PyArrayObject*);
    size_t get_ncol() const;
    size_t get_nrow() const;
    bool is_data_integer() const;

    void fill_row(size_t, double*);
    const IntegerMatrix& get_raw_int() const;
    const NumericMatrix& get_raw_dbl() const;

private:
    bool is_integer;
    size_t nrow;
    size_t ncol;
    NumericMatrix dmat;
    IntegerMatrix imat;
};

/// helper function to check whether an array contains int or double
/// true for integer, false for double, raise an exception if neither
bool is_integer_array(PyArrayObject* arr);

compressed_matrix check_CM_dims(PyArrayObject*, size_t, size_t, const char*, const char*);

NumericMatrix check_design_matrix(PyArrayObject*, size_t);

bool check_logical_scalar(PyObject*, const char*);

long check_integer_scalar(PyObject*, const char*);

double check_numeric_scalar(PyObject*, const char*);

#endif
