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
