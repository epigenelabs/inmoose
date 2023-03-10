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

// This file is based on the file 'src/utils.h' of the Bioconductor edgeR package (version 3.38.4).


#ifndef UTILS_H
#define UTILS_H
//#define DEBUG

#include <limits>
#ifdef DEBUG
#include <iostream>
#endif

#ifndef USE_FC_LEN_T
#define USE_FC_LEN_T
#endif
#ifndef FCONE
#define FCONE
#endif

#include <Python.h>
#include <numpy/arrayobject.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
#include <sstream>
#include <stdexcept>

#include "initialize_levenberg.h"
#include "edgepy_cpp.h"

/* Py-accessible functions are defined in file __init__.pxd */

/* Other utility functions and values */

const double low_value=std::pow(10.0, -10.0), log_low_value=std::log(low_value);

const double LNtwo=std::log(2), one_million=1000000, LNmillion=std::log(one_million);

const double pos_inf = std::numeric_limits<double>::infinity();
const double neg_inf = std::numeric_limits<double>::infinity() * -1;

#endif
