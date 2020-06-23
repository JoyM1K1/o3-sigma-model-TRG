//
// Created by Joy on 2020/06/05.
//
#include "../include/TRG.hpp"
#include <mkl.h>
#include <iostream>
#include <iomanip>
#include <cmath>

#define REP(i, N) for (int i = 0; i < (N); ++i)
#define REP4(i, j, k, l, N) REP(i, N) REP(j, N) REP(k, N) REP(l, N)

using std::cin;
using std::cout;
using std::cerr;
using std::string;

template<typename T>
void print_matrix(T *matrix, MKL_INT m, MKL_INT n, MKL_INT lda, const string &message) {
    cout << '\n' << message << '\n';
    REP(i, m) {
        REP(j, n) {
            cout << std::scientific << std::setprecision(5) << (matrix[i * lda + j] >= 0 ? " " : "")
                 << matrix[i * lda + j]
                 << ' ';
        }
        cout << '\n';
    }
    cout << '\n';
}

void TRG::solver(const int &D_cut, Tensor &T) {
    const int D = T.GetDx(); // same as T.GetDy()
    MKL_INT D_new = std::min(D * D, D_cut);
    auto M = new Tensor(D);
    REP4(i, j, k, l, D) (*M)(i, j, k, l) = T(i, j, k, l); // M(ij)(kl)
    auto sigma = new double[D * D];
    auto U1 = new double[D * D * D * D], VT1 = new double[D * D * D * D];
    auto U2 = new double[D * D * D * D], VT2 = new double[D * D * D * D];
    auto superb = new double[D * D - 1];
    MKL_INT info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', D * D, D * D, (*M).GetMatrix(), D * D, sigma, U1, D * D, VT1, D * D,
                                  superb); // M = U * sigma * VT
    if (info > 0) {
        cerr << "The algorithm computing SVD failed to converge.\n";
        exit(1);
    }
    REP(k, D_new) {
        double s = std::sqrt(sigma[k]);
        REP(i, D) {
            REP(j, D) {
                U1[D * D * D * i + D * D * j + k] *= s; // S1[i][j][k]
                VT1[D * D * k + D * i + j] *= s; // S3[i][j][k]
            }
        }
    }
    REP4(i, j, k, l, D)  (*M)(j, k, l, i) = T(i, j, k, l); // M(jk)(li)
    info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', D * D, D * D, (*M).GetMatrix(), D * D, sigma, U2, D * D, VT2, D * D, superb);
    if (info > 0) {
        cerr << "The algorithm computing SVD failed to converge.\n";
        exit(1);
    }
    delete [] superb;
    delete M;
    REP(k, D_new) {
        double s = std::sqrt(sigma[k]);
        REP(i, D) {
            REP(j, D) {
                U2[D * D * D * i + D * D * j + k] *= s; // S2[i][j][k]
                VT2[D * D * k + D * i + j] *= s; // S4[i][j][k]
            }
        }
    }
    delete [] sigma;

    Tensor X12(D_new, D), X34(D_new, D);
    REP(i, D_new) {
        REP(j, D_new) {
            REP(b, D) {
                REP(d, D) {
                    double sum1 = 0, sum2 = 0;
                    REP(a, D) {
                        sum1 += U1[D * D * D * a + D * D * d + i] * U2[D * D * D * b + D * D * a + j]; // S1[a][d][i] * S2[b][a][j]
                        sum2 += VT1[D * D * i + D * a + b] * VT2[D * D * j + D * d + a]; // S3[a][b][i] * S4[d][a][j]
                    }
                    X12(i, b, j, d) = sum1;
                    X34(i, b, j, d) = sum2;
                }
            }
        }
    }
    delete [] U1;
    delete [] U2;
    delete [] VT1;
    delete [] VT2;

    // 更新
    T.UpdateDx(D_new);
    T.UpdateDy(D_new);

    REP4(i, j, k, l, D_new) {
                    double sum = 0;
                    REP(b, D)REP(d, D) sum += X12(k, b, l, d) * X34(i, b, j, d);
                    T(i, j, k, l) = sum;
                }
}