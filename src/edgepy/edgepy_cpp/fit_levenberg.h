#ifndef FIT_LEVENBERG_H
#define FIT_LEVENBERG_H

#include "utils.h"

PyObject* fit_levenberg (PyArrayObject* y, PyArrayObject* offset, PyArrayObject* disp, PyArrayObject* weights, PyArrayObject* design, PyArrayObject* beta, double tol, long maxit);

#endif
