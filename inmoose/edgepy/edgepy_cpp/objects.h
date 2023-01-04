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
#include <iterator>
#ifndef OBJECTS_H
#define OBJECTS_H

void init_numpy_c_api();

/// This class allows to iterate through a line or a column of a matrix.
/// To avoid undefined behavior if the current pointer goes out-of-bound,
/// it relies on a count to the end as a sentinel.
/// Two iterators compare equal if they have the same begin, count and incr.
///
/// (for a column-wise matrix)
/// Iterator over a column is obtained by setting count=nrow and incr=1
/// Iterator over a line is obtained by setting count=ncol and incr=nrow
template<class T>
class MatrixRowColIterator
{
public:
  using difference_type = typename std::iterator_traits<T*>::difference_type;
  using value_type = typename std::iterator_traits<T*>::value_type;
  using pointer = typename std::iterator_traits<T*>::pointer;
  using reference = typename std::iterator_traits<T*>::reference;
  using iterator_category = std::forward_iterator_tag;

  explicit MatrixRowColIterator(const T* begin, T* current, size_t count, size_t incr)
    : begin(begin)
    , incr(incr)
    , count(count)
    , current(current)
  {}

  const T* operator->() const
  {
    return current;
  }
  const T& operator*() const
  {
    return *current;
  }
  T& operator*()
  {
    return *current;
  }

  bool operator==(const MatrixRowColIterator& other) const
  {
    return count == other.count && begin == other.begin && incr == other.incr;
  }
  bool operator!=(const MatrixRowColIterator& other) const
  {
    return !this->operator==(other);
  }

  MatrixRowColIterator& operator++()
  {
    current += incr;
    count -= 1;
    return *this;
  }
  MatrixRowColIterator operator++(int)
  {
    auto res = *this;
    current += incr;
    count -= 1;
    return res;
  }

private:
  const T* const begin;
  const size_t incr;
  size_t count;

  T* current;
};

template<class T>
class ConstMatrixRowColIterator
{
public:
  using difference_type = typename std::iterator_traits<const T*>::difference_type;
  using value_type = typename std::iterator_traits<const T*>::value_type;
  using pointer = typename std::iterator_traits<const T*>::pointer;
  using reference = typename std::iterator_traits<const T*>::reference;
  using iterator_category = std::forward_iterator_tag;

  explicit ConstMatrixRowColIterator(const T* begin, const T* current, size_t count, size_t incr)
    : begin(begin)
    , incr(incr)
    , count(count)
    , current(current)
  {}

  const T* operator->() const
  {
    return current;
  }
  const T& operator*() const
  {
    return *current;
  }

  bool operator==(const ConstMatrixRowColIterator& other) const
  {
    return count == other.count && begin == other.begin && incr == other.incr;
  }
  bool operator!=(const ConstMatrixRowColIterator& other) const
  {
    return !this->operator==(other);
  }

  ConstMatrixRowColIterator& operator++()
  {
    current += incr;
    count -= 1;
    return *this;
  }
  ConstMatrixRowColIterator operator++(int)
  {
    auto res = *this;
    current += incr;
    count -= 1;
    return res;
  }

private:
  const T* const begin;
  const size_t incr;
  size_t count;

  const T* current;
};

// mimics Matrix class from Rcpp
// /!\ R Matric are stored column-wise (i.e. Fortran order)
// this layout is important to respect, for the sake of the C++ code inherited from edgeR
template<class T>
class Matrix {
public:
  /// constructor from a numpy matrix
  /// constructor may raise exception if:
  ///   - shape of the argument does not correspond to a matrix
  ///   - datatype stored in the argument does not match T
  explicit Matrix(PyArrayObject*);
  /// constructor from dimensions
  explicit Matrix(size_t nrow, size_t ncol);
  ~Matrix();

  Matrix(const Matrix&) = delete;
  Matrix& operator=(const Matrix&) = delete;
  Matrix(Matrix&&) = default;
  Matrix& operator=(Matrix&&) = default;

  const T* begin() const;
  const T* end() const;
  T* begin();
  T* end();

  const T& at(size_t i, size_t j) const
  {
    return data()[i + nrow*j];
  }

  size_t get_ncol() const;
  size_t get_nrow() const;

  class MatrixRow
  {
  public:
    explicit MatrixRow(Matrix& mat, size_t index)
      : mat(mat)
      , index(index)
    {}

    MatrixRowColIterator<T> begin()
    {
      return MatrixRowColIterator<T>(
          mat.begin(),
          mat.begin() + index,
          mat.get_ncol(),
          mat.get_nrow());
    }
    MatrixRowColIterator<T> end()
    {
      return MatrixRowColIterator<T>(
          mat.begin(),
          mat.end(),
          0,
          mat.get_nrow());
    }

  private:
    Matrix& mat;
    size_t index;
  };

  class ConstMatrixRow
  {
  public:
    explicit ConstMatrixRow(const Matrix& mat, size_t index)
      : mat(mat)
      , index(index)
    {}

    ConstMatrixRowColIterator<T> begin() const
    {
      return ConstMatrixRowColIterator<T>(
          mat.begin(),
          mat.begin() + index,
          mat.get_ncol(),
          mat.get_nrow());
    }
    ConstMatrixRowColIterator<T> end() const
    {
      return ConstMatrixRowColIterator<T>(
          mat.begin(),
          mat.end(),
          0,
          mat.get_nrow());
    }

    const T& operator[](size_t j) const
    {
      return mat.at(index, j);
    }

  private:
    const Matrix& mat;
    size_t index;
  };

  class MatrixColumn
  {
  public:
    explicit MatrixColumn(Matrix& mat, size_t index)
      : mat(mat)
      , index(index)
    {}

    MatrixRowColIterator<T> begin()
    {
      return MatrixRowColIterator<T>(
          mat.begin(),
          mat.begin() + index*mat.get_nrow(),
          mat.get_nrow(),
          1);
    }
    MatrixRowColIterator<T> end()
    {
      return MatrixRowColIterator<T>(
          mat.begin(),
          mat.end(),
          0,
          1);
    }

  private:
    Matrix& mat;
    size_t index;
  };

  class ConstMatrixColumn
  {
  public:
    explicit ConstMatrixColumn(const Matrix& mat, size_t index)
      : mat(mat)
      , index(index)
    {}

    ConstMatrixRowColIterator<T> begin() const
    {
      return ConstMatrixRowColIterator<T>(
          mat.begin(),
          mat.begin() + index*mat.get_nrow(),
          mat.get_nrow(),
          1);
    }
    ConstMatrixRowColIterator<T> end() const
    {
      return ConstMatrixRowColIterator<T>(
          mat.begin(),
          mat.end(),
          0,
          1);
    }

    const T& operator[](size_t j) const
    {
      return mat.at(j, index);
    }

  private:
    const Matrix& mat;
    size_t index;
  };

  MatrixRow row(size_t index)
  {
    return MatrixRow(*this, index);
  }
  MatrixColumn col(size_t index)
  {
    return MatrixColumn(*this, index);
  }
  const ConstMatrixRow row(size_t index) const
  {
    return ConstMatrixRow(*this, index);
  }
  const ConstMatrixColumn col(size_t index) const
  {
    return ConstMatrixColumn(*this, index);
  }

  PyArrayObject* to_ndarray();

private:
  T* data();
  const T* data() const;

  size_t nrow;
  size_t ncol;
  PyArrayObject* obj;
  bool owned;
};

using NumericMatrix = Matrix<double>;
using IntegerMatrix = Matrix<long>;

using NumericVector = std::vector<double>;
using IntegerVector = std::vector<long>;
using LogicalVector = std::vector<char>;

/// This class is supposed to give access to the R-side CompressedMatrix class from C++.
/// Because we currently do not replicate CompressedMatrix in Python, this class is close
/// to being an empty shell.
class compressed_matrix {
public:
    explicit compressed_matrix(PyArrayObject*);
    const double* get_row(size_t) const;
    NumericMatrix::ConstMatrixRow row(size_t index) const;
    size_t get_ncol() const;
    size_t get_nrow() const;

    const double* begin() const { return mat.begin(); }
    const double* end() const { return mat.end(); }

    bool is_row_repeated() const { return false; }
    bool is_col_repeated() const { return false; }
private:
    NumericMatrix mat;
    // helper struct to get a view of a row of the matrix
    mutable std::vector<double> output;
};

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
