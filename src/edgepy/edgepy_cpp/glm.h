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
