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

// This file is based on the file 'src/interpolator.cpp' of the Bioconductor edgeR package (version 3.38.4).


#include "interpolator.h"

/*
 *	Splines a la Forsythe Malcolm and Moler
 *	---------------------------------------
 *	In this case the end-conditions are determined by fitting
 *	cubic polynomials to the first and last 4 points and matching
 *	the third derivitives of the spline at the end-points to the
 *	third derivatives of these cubics at the end-points.
 *
 *	This function is taken verbatim from splines.c in the R stats package.
 *	https://github.com/SurajGupta/r-source/blob/master/src/library/stats/src/splines.c
 *
 *  'n' is the number of points, 'x' is the vector of x-coordinates, 'y' is the
 *  vector of y-coordinates (unchanged and equal to the constant for the interpolating
 *  cubic spline upon output), 'b' is the coefficient degree 1, 'c' is the coefficient
 *  of degree '2', 'd' is the coefficient degree 3.
 */

static
void
fmm_spline(const size_t n, const double *x, const double *y, double *b, double *c, double *d)
{
    size_t nm1, i;
    double t;

    /* Adjustment for 1-based arrays */

    x--; y--; b--; c--; d--;

    if(n < 2) {
	errno = EDOM;
	return;
    }

    if(n < 3) {
	t = (y[2] - y[1]);
	b[1] = t / (x[2]-x[1]);
	b[2] = b[1];
	c[1] = c[2] = d[1] = d[2] = 0.0;
	return;
    }

    nm1 = n - 1;

    /* Set up tridiagonal system */
    /* b = diagonal, d = offdiagonal, c = right hand side */

    d[1] = x[2] - x[1];
    c[2] = (y[2] - y[1])/d[1];/* = +/- Inf	for x[1]=x[2] -- problem? */
    for(i=2 ; i<n ; i++) {
	d[i] = x[i+1] - x[i];
	b[i] = 2.0 * (d[i-1] + d[i]);
	c[i+1] = (y[i+1] - y[i])/d[i];
	c[i] = c[i+1] - c[i];
    }

    /* End conditions. */
    /* Third derivatives at x[0] and x[n-1] obtained */
    /* from divided differences */

    b[1] = -d[1];
    b[n] = -d[nm1];
    c[1] = c[n] = 0.0;
    if(n > 3) {
	c[1] = c[3]/(x[4]-x[2]) - c[2]/(x[3]-x[1]);
	c[n] = c[nm1]/(x[n] - x[n-2]) - c[n-2]/(x[nm1]-x[n-3]);
	c[1] = c[1]*d[1]*d[1]/(x[4]-x[1]);
	c[n] = -c[n]*d[nm1]*d[nm1]/(x[n]-x[n-3]);
    }

    /* Gaussian elimination */

    for(i=2 ; i<=n ; i++) {
	t = d[i-1]/b[i-1];
	b[i] = b[i] - t*d[i-1];
	c[i] = c[i] - t*c[i-1];
    }

    /* Backward substitution */

    c[n] = c[n]/b[n];
    for(i=nm1 ; i>=1 ; i--)
	c[i] = (c[i]-d[i]*c[i+1])/b[i];

    /* c[i] is now the sigma[i-1] of the text */
    /* Compute polynomial coefficients */

    b[n] = (y[n] - y[n-1])/d[n-1] + d[n-1]*(c[n-1]+ 2.0*c[n]);
    for(i=1 ; i<=nm1 ; i++) {
	b[i] = (y[i+1]-y[i])/d[i] - d[i]*(c[i+1]+2.0*c[i]);
	d[i] = (c[i+1]-c[i])/d[i];
	c[i] = 3.0*c[i];
    }
    c[n] = 3.0*c[n];
    d[n] = d[nm1];
    return;
}


struct solution {
	double sol1, sol2;
	bool solvable;
};

solution quad_solver (const double& a, const double& b, const double& c) {
	double back=b*b-4*a*c;
	solution cur_sol;
	if (back<0) {
		cur_sol.solvable=false;
		return(cur_sol);
	}
	double front=-b/(2*a);
	back=std::sqrt(back)/(2*a);
	cur_sol.sol1=front-back;
	cur_sol.sol2=front+back;
	cur_sol.solvable=true;
	return(cur_sol);
}

/************************************
 *
 * It fits the spline and grabs the coefficients of each segment.
 * It then calculates the maxima at the segments neighbouring
 * the maximally highest point. This avoids NR optimisation
 * as well as the need to call R's splinefun's from within C.
 *
 ***********************************/

interpolator::interpolator(const size_t& n) : npts(n), b(npts), c(npts), d(npts) {
	if (npts<2) { throw std::runtime_error("must have at least two points for interpolation"); }
    return;
}

double interpolator::find_max (const double*x, const double* y) {
    double maxed=-1;
	size_t maxed_at=-1;
	for (size_t i=0; i<npts; ++i) {
	// Getting a good initial guess for the MLE.
	    if (maxed_at == (size_t)-1 || y[i] > maxed) {
           	maxed=y[i];
           	maxed_at=i;
 	   	}
	}
    double x_max=x[maxed_at];
    fmm_spline(npts, x, y, b.data(), c.data(), d.data());

	// First we have a look at the segment on the left and see if it contains the maximum.
    if (maxed_at>0) {
        const double& ld=d[maxed_at-1];
        const double& lc=c[maxed_at-1];
        const double& lb=b[maxed_at-1];
        solution sol_left=quad_solver(3*ld, 2*lc, lb);
        if (sol_left.solvable) {
            /* Using the solution with the maximum (not minimum). If the curve is mostly increasing, the
             * maximal point is located at the smaller solution (i.e. sol1 for a>0). If the curve is mostly
             * decreasing, the maximal is located at the larger solution (i.e., sol1 for a<0).
             */
            const double& chosen_sol=sol_left.sol1;

            /* The spline coefficients are designed such that 'x' in 'y + b*x + c*x^2 + d*x^3' is
             * equal to 'x_t - x_l' where 'x_l' is the left limit of that spline segment and 'x_t'
             * is where you want to get an interpolated value. This is necessary in 'splinefun' to
             * ensure that you get 'y' (i.e. the original data point) when 'x=0'. For our purposes,
             * the actual MLE corresponds to 'x_t' and is equal to 'solution + x_0'.
             */
            if (chosen_sol > 0 && chosen_sol < x[maxed_at]-x[maxed_at-1]) {
                double temp=((ld*chosen_sol+lc)*chosen_sol+lb)*chosen_sol+y[maxed_at-1];
                if (temp > maxed) {
                    maxed=temp;
                    x_max=chosen_sol+x[maxed_at-1];
                }
            }
        }
    }

	// Repeating for the segment on the right.
    if (maxed_at<npts-1) {
        const double& rd=d[maxed_at];
        const double& rc=c[maxed_at];
        const double& rb=b[maxed_at];
        solution sol_right=quad_solver(3*rd, 2*rc, rb);
        if (sol_right.solvable) {
            const double& chosen_sol=sol_right.sol1; // see arguments above.
            if (chosen_sol > 0 && chosen_sol < x[maxed_at+1]-x[maxed_at]) {
                double temp=((rd*chosen_sol+rc)*chosen_sol+rb)*chosen_sol+y[maxed_at];
                if (temp>maxed) {
                    maxed=temp;
                    x_max=chosen_sol+x[maxed_at];
                }
            }
        }
    }

	return x_max;
}


