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

/* Py-accessible functions are defined in file edgepy.pxd */

/* Other utility functions and values */

const double low_value=std::pow(10.0, -10.0), log_low_value=std::log(low_value);

const double LNtwo=std::log(2), one_million=1000000, LNmillion=std::log(one_million);

const double pos_inf = std::numeric_limits<double>::infinity();
const double neg_inf = std::numeric_limits<double>::infinity() * -1;

#endif
