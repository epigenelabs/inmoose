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

#include "matrix.h"

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
    std::cerr << "[MATRIX] refcount = " << Py_REFCNT(obj) << std::endl;
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


template class Matrix<double>;
template class Matrix<long>;
