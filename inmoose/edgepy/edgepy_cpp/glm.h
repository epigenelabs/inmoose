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

// This file is based on the file 'src/glm.h' of the Bioconductor edgeR package (version 3.38.4).


#ifndef GLM_H
#define GLM_H

#include "utils.h"

std::pair<double,bool> glm_one_group(size_t, const double*, const double*,
        const double*, const double*, long, double, double);

void compute_xtwx(int, int, const double*, const double*, double*);

class glm_levenberg {
public:
	glm_levenberg(int, int, const double*, long, double);
	int fit(const double*, const double*, const double*, const double*, double*, double*);

	const bool& is_failure() const;
	const long& get_iterations()  const;
	const double& get_deviance() const;
private:
	const int nlibs;
	const int ncoefs;
	const long maxit;
	const double tolerance;

    const double* design;
    std::vector<double> working_weights, deriv, xtwx, xtwx_copy, dl, dbeta;
    int info;

    std::vector<double> mu_new, beta_new;
	double dev;
	long iter;
	bool failed;

	double nb_deviance(const double*, const double*, const double*, const double*) const;
	void autofill(const double*, const double*, double*);
};

double compute_unit_nb_deviance(double, double, double);

class adj_coxreid {
public:
	adj_coxreid(int, int, const double*);
	std::pair<double, bool> compute(const double* wptr);
private:
	const int ncoefs, nlibs;
    const double* design;
    std::vector<double> xtwx, work;
    std::vector<int> pivots;
    int info, lwork;
};

#endif
