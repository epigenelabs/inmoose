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

// This file is based on the file 'src/objects.cpp' of the Bioconductor edgeR package (version 3.38.4).


#include "objects.h"
#include "numpy/ndarraytypes.h"
#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "edgepy_cpp.h"

// debug
#include <iostream>

void
init_numpy_c_api()
{
  auto f = []() {
    if (!PyArray_API)
    {
      import_array();
    }
    return NULL;
  };
  f();
}

template<class T>
struct NPY
{};
template<>
struct NPY<long>
{
  static_assert(sizeof(long) == 8, "long is not 64-bit!");
  static constexpr int typenum = NPY_INT64;
};
template<>
struct NPY<double>
{
  static_assert(sizeof(double) == 8, "double is not 64-bit!");
  static_assert(std::numeric_limits<double>::is_iec559, "double does not comply with IEEE 754");
  static constexpr int typenum = NPY_DOUBLE;
};

template<class T>
Matrix<T>::Matrix(PyArrayObject* incoming)
  : obj(incoming), owned(false)
{
  if (obj == nullptr)
  {
    nrow = 0;
    ncol = 0;
    return;
  }
  auto ndims = PyArray_NDIM(obj);
  if (ndims != 2)
  {
    throw std::runtime_error("Matrix should have 2 dimensions");
  }
  auto* truedims = PyArray_SHAPE(obj);
  assert(truedims != nullptr);
  nrow=truedims[0];
  ncol=truedims[1];

  if (!PyArray_ISFARRAY(obj))
  {
    PyErr_SetString(PyExc_RuntimeError, "Matrix should be Fortran style");
    //throw std::runtime_error("Matrix should be Fortran style");
  }
  // check type compatibility
  if (PyArray_TYPE(obj) != NPY<T>::typenum)
  {
    PyErr_SetString(PyExc_RuntimeError, "Datatypes do not agree");
  }
}

template<class T>
Matrix<T>::Matrix(size_t nrow, size_t ncol)
  : nrow(nrow), ncol(ncol), owned(true)
{
  npy_intp dims[2] = { (npy_intp)nrow, (npy_intp)ncol };
  obj = (PyArrayObject*) PyArray_Zeros(2, dims, PyArray_DescrFromType(NPY<T>::typenum), 1);
}

template<class T>
Matrix<T>::~Matrix()
{
  if (owned)
  {
    std::cerr << "[MATRIX] refcount = " << PyArray_REFCOUNT(obj) << std::endl;
  }
}

template<class T>
size_t
Matrix<T>::get_nrow() const
{
  return nrow;
}

template<class T>
size_t
Matrix<T>::get_ncol() const
{
  return ncol;
}

template<class T>
const T*
Matrix<T>::begin() const
{
  return data();
}

template<class T>
const T*
Matrix<T>::end() const
{
  return data() + nrow*ncol;
}

template<class T>
T*
Matrix<T>::begin()
{
  return data();
}

template<class T>
T*
Matrix<T>::end()
{
  return data() + nrow*ncol;
}

template<class T>
PyArrayObject*
Matrix<T>::to_ndarray()
{
  owned = false;
  PyArrayObject* res = obj;
  obj = nullptr;
  return res;
}

template<class T>
const T*
Matrix<T>::data() const
{
  return static_cast<const T*>(PyArray_DATA(obj));
}

template<class T>
T*
Matrix<T>::data()
{
  return static_cast<T*>(PyArray_DATA(obj));
}

compressed_matrix::compressed_matrix(PyArrayObject* incoming)
  : mat(incoming)
{
  // Preallocate output vector
  output.resize(mat.get_ncol());
}

NumericMatrix::ConstMatrixRow compressed_matrix::row(size_t index) const
{
  return mat.row(index);
}

const double* compressed_matrix::get_row(size_t index) const
{
  const auto nrow = mat.get_nrow();
  if (index >= nrow)
  {
    throw std::runtime_error("requested row index out of range");
  }
  auto cmat = mat.row(index);
  std::copy(cmat.begin(), cmat.end(), output.begin());
  return output.data();
}

size_t
compressed_matrix::get_nrow() const
{
  return mat.get_nrow();
}

size_t
compressed_matrix::get_ncol() const
{
  return mat.get_ncol();
}

// helper function to check whether an array contains int or double
bool
is_integer_array(PyArrayObject* arr)
{
  if (PyArray_TYPE(arr) == NPY<long>::typenum)
  {
    return true;
  }
  else if (PyArray_TYPE(arr) == NPY<double>::typenum)
  {
    return false;
  }
  else
  {
    PyErr_SetString(PyExc_RuntimeError, "array dtype is neither 'int' nor 'double'");
    return false;
  }
}

/* Methods for any numeric matrix */

any_numeric_matrix::any_numeric_matrix(PyArrayObject* incoming)
  : is_integer(is_integer_array(incoming))
  , dmat(nullptr)
  , imat(nullptr)
{
  if (is_integer)
  {
    imat = IntegerMatrix(incoming);
    nrow = imat.get_nrow();
    ncol = imat.get_ncol();
  }
  else
  {
    dmat = NumericMatrix(incoming);
    nrow = dmat.get_nrow();
    ncol = dmat.get_ncol();
  }
}

void any_numeric_matrix::fill_row(size_t index, double* ptr) {
    if (is_integer) {
        auto current=imat.row(index);
        std::copy(current.begin(), current.end(), ptr);
    } else {
        auto current=dmat.row(index);
        std::copy(current.begin(), current.end(), ptr);;
    }
    return;
}

bool any_numeric_matrix::is_data_integer () const {
    return is_integer;
}

const IntegerMatrix& any_numeric_matrix::get_raw_int() const {
    return imat;
}

const NumericMatrix& any_numeric_matrix::get_raw_dbl() const {
    return dmat;
}

size_t
any_numeric_matrix::get_ncol() const
{
  return ncol;
}

size_t
any_numeric_matrix::get_nrow() const
{
  return nrow;
}

/* Methods to check the dimensions of any object. */

compressed_matrix
check_CM_dims(PyArrayObject* incoming, size_t nrow, size_t ncol, const char * current, const char * ref)
{
  compressed_matrix out(incoming);
  if (out.get_nrow() != nrow || out.get_ncol() != ncol)
  {
    std::stringstream err;
    err << current << " and " << ref << " matrices do not have the same dimensions";
    throw std::runtime_error(err.str().c_str());
  }
  return out;
}

NumericMatrix check_design_matrix(PyArrayObject* design, size_t nlibs)
{
  NumericMatrix X(design);
  if (X.get_nrow() != nlibs)
  {
    throw std::runtime_error("number of rows in the design matrix should be equal to the number of libraries");
  }
  return X;
}

bool
check_logical_scalar(PyObject* obj, const char* thing)
{
  if (!PyBool_Check(obj))
  {
    std::stringstream err;
    err << "expected boolean for the " << thing;
    throw std::runtime_error(err.str().c_str());
  }
  return obj == Py_True;
}

long
check_integer_scalar(PyObject* obj, const char* thing)
{
  if (!PyLong_Check(obj))
  {
    std::stringstream err;
    err << "expected integer for the " << thing;
    throw std::runtime_error(err.str().c_str());
  }
  long res = PyLong_AsLong(obj);
  if (res == -1L && PyErr_Occurred())
  {
    throw std::runtime_error("PyErr occurred");
  }
  return res;
}

double
check_numeric_scalar(PyObject* obj, const char* thing)
{
  if (!PyFloat_Check(obj))
  {
    std::stringstream err;
    err << "expected float for the " << thing;
    throw std::runtime_error(err.str().c_str());
  }
  double res = PyFloat_AsDouble(obj);
  if (res == -1. && PyErr_Occurred())
  {
    throw std::runtime_error("PyErr occurred");
  }
  return res;
}

template class Matrix<double>;
template class Matrix<long>;
