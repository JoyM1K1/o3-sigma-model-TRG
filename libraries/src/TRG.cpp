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

void TRG::solver(int &D, const int &D_cut, std::vector<std::vector<std::vector<std::vector<double>>>> &T) {
    MKL_INT D_new = std::min(D * D, D_cut);
    double M[D * D * D * D];
    REP(i, D * D * D * D) {
        M[i] = 0;
    }
    REP4(i, j, k, l, std::min(D, D_cut)) {
                    M[l + D * k + D * D * j + D * D * D * i] = T[i][j][k][l]; // M(ij)(kl)
                }
    double sigma[D * D];
    double U[D * D * D * D], VT[D * D * D * D];
    double S1[D][D][D_new], S2[D][D][D_new], S3[D][D][D_new], S4[D][D][D_new];
    double superb[D * D];
    MKL_INT info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', D * D, D * D, M, D * D, sigma, U, D * D, VT, D * D,
                                  superb); // M = U * sigma * VT
    if (info > 0) {
        cerr << "The algorithm computing SVD failed to converge.\n";
        exit(1);
    }
    if (false) {
        print_matrix(sigma, 1, D * D, D * D, "sigma");
        print_matrix(U, D * D, D * D, D * D, "U");
//            print_matrix(VT, D * D, D * D, D * D, "VT");
//            double US[D * D * D * D];
//            REP4(i, j, k, l, D) {
//                            US[i + D * j + D * D * k + D * D * D * l] =
//                                    U[i + D * j + D * D * k + D * D * D * l] * sigma[i + D * j];
//                        }
//            double USVH[D * D * D * D];
//            REP(i, D * D * D * D) {
//                USVH[i] = 0;
//            }
//            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, D * D, D * D, D * D, 1, US, D * D, VT, D * D, 0,
//                        USVH, D * D);
//            print_matrix(USVH, D * D, D * D, D * D, "U * sigma * VT");
    }
    REP(k, D_new) {
        double s = std::sqrt(sigma[k]);
        REP(i, D) {
            REP(j, D) {
                S1[i][j][k] = s * U[k + D * D * j + D * D * D * i];
                S3[i][j][k] = s * VT[j + D * i + D * D * k];
            }
        }
    }
    REP(i, D * D * D * D) {
        M[i] = 0;
    }
    REP4(i, j, k, l, std::min(D, D_cut)) {
                    M[i + D * l + D * D * k + D * D * D * j] = T[i][j][k][l]; // M(jk)(li)
                }
    info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', D * D, D * D, M, D * D, sigma, U, D * D, VT, D * D, superb);
    if (info > 0) {
        cerr << "The algorithm computing SVD failed to converge.\n";
        exit(1);
    }
    REP(k, D_new) {
        double s = std::sqrt(sigma[k]);
        REP(i, D) {
            REP(j, D) {
                S2[i][j][k] = s * U[k + D * D * j + D * D * D * i];
                S4[i][j][k] = s * VT[j + D * i + D * D * k];
            }
        }
    }

    double X12[D_new][D_new][D][D], X34[D_new][D_new][D][D];
    REP(i, D_new) {
        REP(j, D_new) {
            REP(b, D) {
                REP(d, D) {
                    X12[i][j][b][d] = 0;
                    X34[i][j][b][d] = 0;
                    REP(a, D) {
                        X12[i][j][b][d] += S1[a][d][i] * S2[b][a][j];
                        X34[i][j][b][d] += S3[a][b][i] * S4[d][a][j];
                    }
                }
            }
        }
    }

    REP4(i, j, k, l, D_new) {
                    T[i][j][k][l] = 0;
                    REP(b, D) {
                        REP(d, D) {
                            T[i][j][k][l] += X12[k][l][b][d] * X34[i][j][b][d];
                        }
                    }
                }

    // 更新
    D = D_new;
}