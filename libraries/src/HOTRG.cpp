//
// Created by Joy on 2020/06/09.
//

#include "../include/HOTRG.hpp"
#include <mkl.h>
#include <iostream>
#include <iomanip>
#include <cmath>

#define REP(i, N) for (int i = 0; i < (N); ++i)
#define REP4(i, j, k, l, N) REP(i, N) REP(j, N) REP(k, N) REP(l, N)

using std::cout;
using std::cerr;

template<typename T>
void print_matrix(const int m, const int n, const T *M) {
    REP(i, std::min(m, 10)) {
        REP(j, std::min(n, 10)) {
            const T component = M[n * i + j];
            cout << std::scientific << std::setprecision(5) << (component < 0 ? "" : " ") << component << ' ';
        }
        cout << '\n';
    }
}

void HOTRG::contractionX(const int &Dx, int &Dy, const int &D_cut,
                         std::vector<std::vector<std::vector<std::vector<double>>>> &T) {
    double M[Dx * Dx * Dy * Dy * Dy * Dy];
    double sigma[Dy * Dy];
    double U_U[Dy * Dy * Dy * Dy], U_D[Dy * Dy * Dy * Dy], VT[1];
    double superb[Dy * Dy - 1];
    double epsilon_1 = 0, epsilon_2 = 0;
    REP(i1, Dx)REP(j1, Dy)REP(l1, Dy)REP(j2, Dy)REP(k2, Dx)REP(l2, Dy) {
                            double sum = 0;
                            REP(a, Dx) sum += T[i1][j1][a][l1] * T[a][j2][k2][l2];
                            M[Dx * Dx * Dy * Dy * Dy * j1 + Dx * Dx * Dy * Dy * j2 + Dx * Dy * Dy * i1 + Dy * Dy * k2 +
                              Dy * l1 + l2] = sum;
                        }
    MKL_INT info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'N', Dy * Dy, Dx * Dx * Dy * Dy, M, Dx * Dx * Dy * Dy, sigma,
                                  U_U,
                                  Dy * Dy, VT, 1, superb);
    if (info > 0) {
        cerr << "The algorithm computing SVD failed to converge.\n";
        exit(1);
    }
    for (int i = D_cut; i < Dy * Dy; ++i) {
        const double s = sigma[i];
        epsilon_1 += s * s;
    }
    if (epsilon_1 != 0) {
        REP(i1, Dx)REP(j1, Dy)REP(l1, Dy)REP(j2, Dy)REP(k2, Dx)REP(l2, Dy) {
                                double sum = 0;
                                REP(a, Dx) sum += T[i1][j1][a][l1] * T[a][j2][k2][l2];
                                M[Dx * Dx * Dy * Dy * Dy * l1 + Dx * Dx * Dy * Dy * l2 + Dx * Dy * Dy * i1 +
                                  Dy * Dy * k2 +
                                  Dy * j1 + j2] = sum;
                            }
        info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'N', Dy * Dy, Dx * Dx * Dy * Dy, M, Dx * Dx * Dy * Dy, sigma, U_D,
                              Dy * Dy, VT, 1, superb);
        if (info > 0) {
            cerr << "The algorithm computing SVD failed to converge.\n";
            exit(1);
        }
        for (int i = D_cut; i < Dx * Dx; ++i) {
            const double s = sigma[i];
            epsilon_2 += s * s;
        }
    }
    REP(i1, Dx)REP(j1, Dy)REP(l1, Dy)REP(j2, Dy)REP(k2, Dx)REP(l2, Dy) {
                            double sum = 0;
                            REP(a, Dx) sum += T[i1][j1][a][l1] * T[a][j2][k2][l2];
                            M[Dx * Dx * Dy * Dy * Dy * j1 + Dx * Dx * Dy * Dy * j2 + Dx * Dy * Dy * i1 + Dy * Dy * k2 +
                              Dy * l1 + l2] = sum;
                        }
    MKL_INT Dy_new = std::min(Dy * Dy, D_cut);
    double *U = (epsilon_1 == 0 || epsilon_1 < epsilon_2) ? U_D : U_U;
    double temp[Dy_new * Dy * Dy * Dx * Dx];
    REP(i, Dx)REP(k, Dx)REP(j, Dy_new)REP(l, Dy * Dy) {
                    double sum = 0;
                    REP(p, Dy * Dy) sum += U[Dy * Dy * p + j] *
                                           M[Dx * Dx * Dy * Dy * p + Dx * Dy * Dy * i + Dy * Dy * k + l];
                    temp[Dx * Dx * Dy * Dy * j + Dx * Dy * Dy * i + Dy * Dy * k + l] = sum;
                }
    REP(i, Dx)REP(k, Dx)REP(j, Dy_new)REP(l, Dy_new) {
                    double sum = 0;
                    REP(p, Dy * Dy) sum += U[Dy * Dy * p + l] *
                                           temp[Dx * Dx * Dy * Dy * j + Dx * Dy * Dy * i + Dy * Dy * k + p];
                    T[i][j][k][l] = sum;
                }
    Dy = Dy_new;
}

void HOTRG::contractionY(int &Dx, const int &Dy, const int &D_cut,
                         std::vector<std::vector<std::vector<std::vector<double>>>> &T) {
    double M[Dx * Dx * Dx * Dx * Dy * Dy];
    double sigma[Dx * Dx];
    double U_R[Dx * Dx * Dx * Dx], U_L[Dx * Dx * Dx * Dx], VT[1];
    double superb[Dx * Dx - 1];
    double epsilon_1 = 0, epsilon_2 = 0;
    REP(i1, Dx)REP(j1, Dy)REP(k1, Dx)REP(i2, Dx)REP(k2, Dx)REP(l2, Dy) {
                            double sum = 0;
                            REP(a, Dy) sum += T[i1][j1][k1][a] * T[i2][a][k2][l2];
                            M[Dx * Dx * Dx * Dy * Dy * i1 + Dx * Dx * Dy * Dy * i2 + Dx * Dy * Dy * k1 + Dy * Dy * k2 +
                              Dy * j1 + l2] = sum;
                        }
    MKL_INT info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'N', Dx * Dx, Dx * Dx * Dy * Dy, M, Dx * Dx * Dy * Dy, sigma,
                                  U_R,
                                  Dx * Dx, VT, 1, superb);
    if (info > 0) {
        cerr << "The algorithm computing SVD failed to converge.\n";
        exit(1);
    }
    for (int i = D_cut; i < Dx * Dx; ++i) {
        const double s = sigma[i];
        epsilon_1 += s * s;
    }
    if (epsilon_1 != 0) {
        REP(i1, Dx)REP(j1, Dy)REP(k1, Dx)REP(i2, Dx)REP(k2, Dx)REP(l2, Dy) {
                                double sum = 0;
                                REP(a, Dy) sum += T[i1][j1][k1][a] * T[i2][a][k2][l2];
                                M[Dx * Dx * Dx * Dy * Dy * k1 + Dx * Dx * Dy * Dy * k2 + Dx * Dy * Dy * i1 +
                                  Dy * Dy * i2 +
                                  Dy * j1 + l2] = sum;
                            }
        info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'N', Dx * Dx, Dx * Dx * Dy * Dy, M, Dx * Dx * Dy * Dy, sigma, U_L,
                              Dx * Dx,
                              VT,
                              1, superb);
        if (info > 0) {
            cerr << "The algorithm computing SVD failed to converge.\n";
            exit(1);
        }
        for (int i = D_cut; i < Dx * Dx; ++i) {
            const double s = sigma[i];
            epsilon_2 += s * s;
        }
    }
    REP(i1, Dx)REP(j1, Dy)REP(k1, Dx)REP(i2, Dx)REP(k2, Dx)REP(l2, Dy) {
                            double sum = 0;
                            REP(a, Dy) sum += T[i1][j1][k1][a] * T[i2][a][k2][l2];
                            M[Dx * Dx * Dx * Dy * Dy * i1 + Dx * Dx * Dy * Dy * i2 + Dx * Dy * Dy * k1 + Dy * Dy * k2 +
                              Dy * j1 + l2] = sum;
                        }
    MKL_INT Dx_new = std::min(Dx * Dx, D_cut);
    double *U = (epsilon_1 == 0 || epsilon_1 < epsilon_2) ? U_L : U_R;
    double temp[Dx_new * Dx * Dx * Dy * Dy];
    REP(j, Dy)REP(l, Dy)REP(i, Dx_new)REP(k, Dx * Dx) {
                    double sum = 0;
                    REP(p, Dx * Dx) sum += U[Dx * Dx * p + i] * M[Dx * Dx * Dy * Dy * p + Dy * Dy * k + Dy * j + l];
                    temp[Dx * Dx * Dy * Dy * i + Dy * Dy * k + Dy * j + l] = sum;
                }
    REP(j, Dy)REP(l, Dy)REP(i, Dx_new)REP(k, Dx_new) {
                    double sum = 0;
                    REP(p, Dx * Dx) sum += U[Dx * Dx * p + k] * temp[Dx * Dx * Dy * Dy * i + Dy * Dy * p + Dy * j + l];
                    T[i][j][k][l] = sum;
                }
    Dx = Dx_new;
}