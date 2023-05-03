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

// This file is based on the file 'src/R_initialize_levenberg.cpp' of the Bioconductor edgeR package (version 3.38.4).


#include "initialize_levenberg.h"
#include "glm.h"
#include "numpy/ndarraytypes.h"
#include "objects.h"

/* Different initialization methods for the Levenberg coefficients */

void build_QRdecomposition(QRdecomposition*);
void decompose_QRdecomposition(QRdecomposition*);
void solve_QRdecomposition(QRdecomposition*);

QRdecomposition::QRdecomposition(int nrows, int ncoefs, const double* curX) : NR(nrows), NC(ncoefs),
        X(curX), Xcopy(NR*NC), tau(NC), effects(NR), weights(NR),
        lwork_geqrf(-1), lwork_ormqr(-1) {

    build_QRdecomposition(this);
}

// set (or reset) the weights
// NB: the internal weights are the sqrt of the input weights
void QRdecomposition::store_weights(const double* w) {
    if (w==NULL) {
        std::fill(weights.begin(), weights.end(), 1);
    } else {
        for (int row=0; row<NR; ++row) {
            weights[row]=std::sqrt(w[row]);
        }
    }
    return;
}

void QRdecomposition::decompose() {
    auto xcIt=Xcopy.begin();
    std::copy(X, X + Xcopy.size(), xcIt);
    for (int coef=0; coef<NC; ++coef) {
        for (int lib=0; lib<NR; ++lib) {
            (*xcIt)*=weights[lib];
            ++xcIt;
        }
    }

    // At this point, Xcopy contains X with all rows reweighted by `weights`
    // Let us call it WX (for "weighted X")

    decompose_QRdecomposition(this);
    // At this point, Xcopy contains the QR decomposition of WX
    if (info) {
        throw std::runtime_error("QR decomposition failed");
    }
   return;
}

void QRdecomposition::solve(const double * y) {
    for (int row=0; row<NR; ++row) {
        effects[row]=y[row]*weights[row];
    }

    solve_QRdecomposition(this);

    return;
}

PyArrayObject* get_levenberg_start(PyArrayObject* y, PyArrayObject* offset, PyArrayObject* disp, PyArrayObject* weights, PyArrayObject* design, bool use_null) {
    any_numeric_matrix counts(y);
    const size_t num_tags=counts.get_nrow();
    const size_t num_libs=counts.get_ncol();

    NumericMatrix X=check_design_matrix(design, num_libs);
    const size_t num_coefs=X.get_ncol();
    QRdecomposition QR(num_libs, num_coefs, X.begin());

    // Initializing pointers to the assorted features.
    compressed_matrix allo=check_CM_dims(offset, num_tags, num_libs, "offset", "count");
    compressed_matrix alld=check_CM_dims(disp, num_tags, num_libs, "dispersion", "count");
    compressed_matrix allw=check_CM_dims(weights, num_tags, num_libs, "weight", "count");

    // Determining what type of algorithm is to be used.
    bool null_method = use_null;

    NumericMatrix output(num_tags, num_coefs);
    std::vector<double> current(num_libs);
    if (null_method) {
        QR.store_weights(NULL);
        QR.decompose();

        for (size_t tag=0; tag<num_tags; ++tag) {
            counts.fill_row(tag, current.data());
            auto dptr=alld.row(tag);
            auto optr=allo.row(tag);
            auto wptr=allw.row(tag);

            // Computing weighted average of the count:library size ratios.
            double sum_weight=0, sum_exprs=0;
            for (size_t lib=0; lib<num_libs; ++lib) {
                const double curN=std::exp(optr[lib]);
                const double curweight=wptr[lib]*curN/(1 + dptr[lib] * curN);
                sum_exprs += current[lib] * curweight / curN;
                sum_weight += curweight;
            }
            std::fill(current.begin(), current.end(), std::log(sum_exprs/sum_weight));

            // Performing the QR decomposition and taking the solution.
            QR.solve(current.data());
            auto curout=output.row(tag);
            std::copy(QR.effects.begin(), QR.effects.begin()+num_coefs, curout.begin());
        }
    } else {
        const bool weights_are_the_same=allw.is_row_repeated();
        if (weights_are_the_same && num_tags) {
            QR.store_weights(allw.get_row(0));
            QR.decompose();
        }

        // Finding the delta.
        double delta=0;
        if (counts.is_data_integer()) {
            const IntegerMatrix& imat=counts.get_raw_int();
            delta=*std::max_element(imat.begin(), imat.end());
        } else {
            const NumericMatrix& dmat=counts.get_raw_dbl();
            delta=*std::max_element(dmat.begin(), dmat.end());
        }
        delta=std::min(delta, 1.0/6);

        for (size_t tag=0; tag<num_tags; ++tag) {
            if (!weights_are_the_same) {
                QR.store_weights(allw.get_row(tag));
                QR.decompose();
            }
            counts.fill_row(tag, current.data());
            auto optr=allo.row(tag);

            // Computing normalized log-expression values.
            for (size_t lib=0; lib<num_libs; ++lib) {
                current[lib]=std::log(std::max(delta, current[lib])) - optr[lib];
            }

            // Performing the QR decomposition and taking the solution.
            QR.solve(current.data());
            auto curout=output.row(tag);
            std::copy(QR.effects.begin(), QR.effects.begin()+num_coefs, curout.begin());
        }
    }

    return output.to_ndarray();
}
