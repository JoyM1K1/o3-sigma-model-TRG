#include "../include/TRG.hpp"
#include <mkl.h>
#include <iostream>
#include <cmath>

#define REP(i, N) for (int i = 0; i < (N); ++i)
#define REP4(i, j, k, l, N) REP(i, N) REP(j, N) REP(k, N) REP(l, N)

using std::cin;
using std::cout;
using std::cerr;
using std::string;

void TRG::solver(const int &D_cut, Tensor &T) {
    const int D = T.GetDx(); // same as T.GetDy()
    const int D_new = std::min(D * D, D_cut);
    auto M = new Tensor(D);
    REP4(i, j, k, l, D) (*M)(i, j, k, l) = T(i, j, k, l); // M(ij)(kl)
    auto sigma = new double[D * D];
    auto U = new double[D * D * D * D], VT = new double[D * D * D * D];
    auto S1 = new double[D * D * D_new], S3 = new double[D * D * D_new];
    auto superb = new double[D * D - 1];
    MKL_INT info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', D * D, D * D, (*M).GetMatrix(), D * D, sigma, U, D * D, VT, D * D, superb);
    if (info > 0) {
        cerr << "The algorithm computing SVD failed to converge.\n";
        exit(1);
    }
    REP(k, D_new) {
        double s = std::sqrt(sigma[k]);
        REP(i, D)REP(j, D) {
                S1[D * D * k + D * j + i] = U[D * D * D * i + D * D * j + k] * s;
                S3[D * D * k + D * j + i] = VT[D * D * k + D * i + j] * s;
            }
    }
    auto S2 = new double[D * D * D_new], S4 = new double[D * D * D_new];
    REP4(i, j, k, l, D) (*M)(j, k, l, i) = T(i, j, k, l); // M(jk)(li)
    info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', D * D, D * D, (*M).GetMatrix(), D * D, sigma, U, D * D, VT, D * D, superb);
    if (info > 0) {
        cerr << "The algorithm computing SVD failed to converge.\n";
        exit(1);
    }
    delete[] superb;
    delete M;
    REP(k, D_new) {
        double s = std::sqrt(sigma[k]);
        REP(i, D)REP(j, D) {
                S2[D_new * D * j + D_new * i + k] = U[D * D * D * i + D * D * j + k] * s;
                S4[D_new * D * j + D_new * i + k] = VT[D * D * k + D * i + j] * s;
            }
    }
    delete[] U;
    delete[] VT;
    delete[] sigma;

    auto top = new double[D_new * D_new * D * D], bottom = new double[D_new * D_new * D * D];
    auto X = new double[D_new * D_new * D * D];
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, D_new * D, D_new * D, D, 1, S1,
            D, S2, D_new * D, 0, X, D_new * D);
    delete[] S1;
    delete[] S2;
    REP(a, D)REP(b, D)REP(i, D_new)REP(j, D_new) {
                    bottom[D_new * D_new * D * a + D_new * D_new * b + D_new * i + j] = X[D_new * D * D * i + D_new * D * b + D_new * a + j];
                }
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, D_new * D, D_new * D, D, 1, S3,
            D, S4, D_new * D, 0, X, D_new * D);
    delete[] S3;
    delete[] S4;
    REP(a, D)REP(b, D)REP(i, D_new)REP(j, D_new) {
                    top[D * D * D_new * i + D * D * j + D * a + b] = X[D_new * D * D * i + D_new * D * a + D_new * b + j];
                }
    delete[] X;

    // 更新
    T.UpdateDx(D_new);
    T.UpdateDy(D_new);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, D_new * D_new, D_new * D_new, D * D, 1, top, D * D, bottom, D_new * D_new, 0, T.GetMatrix(), D_new * D_new);
    delete[] top;
    delete[] bottom;
}