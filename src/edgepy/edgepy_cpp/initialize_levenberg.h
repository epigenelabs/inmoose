#ifndef INITIALIZE_LEVENBERG_H
#define INITIALIZE_LEVENBERG_H

#include <vector>

class QRdecomposition {
public:
    int NR, NC;
    const double* X;
    std::vector<double> Xcopy, tau, effects, weights, work_geqrf, work_ormqr;
    int lwork_geqrf, lwork_ormqr, info;

    QRdecomposition(int nrows, int ncoefs, const double* curX);
    void store_weights(const double* w);
    void decompose();
    void solve(const double*);
};

#endif
