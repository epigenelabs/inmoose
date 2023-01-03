#ifndef INTERPOLATOR_H
#define INTERPOLATOR_H

#include "utils.h"

/* This class just identifies the global maximum in the interpolating function. */

class interpolator {
public:
	interpolator(const size_t&);
	double find_max(const double* x, const double* y);
private:
	const size_t npts;
    std::vector<double> b, c, d;
};


#endif
