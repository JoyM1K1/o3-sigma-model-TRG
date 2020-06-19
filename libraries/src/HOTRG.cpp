//
// Created by Joy on 2020/06/09.
//

#include "../include/HOTRG.hpp"
#include <mkl.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>
#include <vector>

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

void HOTRG::SVD_X(const int D_cut, Tensor &T, double *U) {
    const int Dx = T.GetDx(), Dy = T.GetDy();
    Tensor M(Dx * Dx, Dy);
    const int m = M.GetDx();
    const int n = M.GetDx() * M.GetDy() * M.GetDy();
    auto *temp = new double [m * m];
    auto *sigma = new double [m];
    auto *superb = new double [m - 1];
    double sum;
    #pragma omp parallel private(sum)
    {
        #pragma omp for schedule(static)
        REP(i1, Dx)
            REP(j1, Dy)
                REP(k1, Dx)
                    REP(i2, Dx)
                        REP(k2, Dx)
                            REP(l2, Dy) {
                                sum = 0;
                                REP(a, Dy) sum += T(i1, j1, k1, a) * T(i2, a, k2, l2);
                                M(Dx * i1 + i2, j1, Dx * k1 + k2, l2) = sum;
                            }
    }
    MKL_INT info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'N', m, n, M.GetMatrix(), n, sigma, U, m, nullptr, 1, superb);
    if (info > 0) {
        cerr << "The algorithm computing SVD failed to converge.\n";
        exit(1);
    }
    double epsilon_1 = 0, epsilon_2 = 0;
    for (int i = D_cut; i < m; ++i) {
        const double s = sigma[i];
        epsilon_1 += s * s;
    }
    if (epsilon_1 != 0) {
        #pragma omp parallel private(sum)
        {
            #pragma omp for schedule(static)
            REP(i1, Dx)
                REP(j1, Dy)
                    REP(k1, Dx)
                        REP(i2, Dx)
                            REP(k2, Dx)
                                REP(l2, Dy) {
                                    sum = 0;
                                    REP(a, Dy) sum += T(i1, j1, k1, a) * T(i2, a, k2, l2);
                                    M(Dx * k1 + k2, j1, Dx * i1 + i2, l2) = sum;
                                }
        }
        info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'N', m, n, M.GetMatrix(), n, sigma, temp, m, nullptr, 1, superb);
        if (info > 0) {
            cerr << "The algorithm computing SVD failed to converge.\n";
            exit(1);
        }
        for (int i = D_cut; i < m; ++i) {
            const double s = sigma[i];
            epsilon_2 += s * s;
        }
        if (epsilon_1 > epsilon_2) {
            REP(i, m * m) U[i] = temp[i];
        }
    }
    delete [] temp;
    delete [] sigma;
    delete [] superb;
}

void HOTRG::SVD_Y(const int D_cut, Tensor &T, double *U) {
    const int Dx = T.GetDx(), Dy = T.GetDy();
    Tensor M(Dy * Dy, Dx);
    const int m = M.GetDx();
    const int n = M.GetDx() * M.GetDy() * M.GetDy();
    auto *temp = new double [m * m];
    auto *sigma = new double [m];
    auto *superb = new double [m - 1];
    double sum;
    #pragma omp parallel private(sum)
    {
        #pragma omp for schedule(static)
        REP(i1, Dx)
            REP(j1, Dy)
                REP(l1, Dy)
                    REP(j2, Dy)
                        REP(k2, Dx)
                            REP(l2, Dy) {
                                sum = 0;
                                REP(a, Dx) sum += T(i1, j1, a, l1) * T(a, j2, k2, l2);
                                M(Dy * j1 + j2, i1, Dy * l1 + l2, k2) = sum;
                            }
    }
    MKL_INT info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'N', m, n, M.GetMatrix(), n, sigma, U, m, nullptr, 1, superb);
    if (info > 0) {
        cerr << "The algorithm computing SVD failed to converge.\n";
        exit(1);
    }
    double epsilon_1 = 0, epsilon_2 = 0;
    for (int i = D_cut; i < m; ++i) {
        const double s = sigma[i];
        epsilon_1 += s * s;
    }
    if (epsilon_1 != 0) {
        #pragma omp parallel private(sum)
        {
            #pragma omp for schedule(static)
            REP(i1, Dx)
                REP(j1, Dy)
                    REP(l1, Dy)
                        REP(j2, Dy)
                            REP(k2, Dx)
                                REP(l2, Dy) {
                                    sum = 0;
                                    REP(a, Dx) sum += T(i1, j1, a, l1) * T(a, j2, k2, l2);
                                    M(Dy * l1 + l2, i1, Dy * j1 + j2, k2) = sum;
                                }
        }
        info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'N', m, n, M.GetMatrix(), n, sigma, temp, m, nullptr, 1, superb);
        if (info > 0) {
            cerr << "The algorithm computing SVD failed to converge.\n";
            exit(1);
        }
        for (int i = D_cut; i < m; ++i) {
            const double s = sigma[i];
            epsilon_2 += s * s;
        }
        if (epsilon_1 > epsilon_2) {
            REP(i, m * m) U[i] = temp[i];
        }
    }
    delete [] temp;
    delete [] sigma;
    delete [] superb;
}

// contraction right tensor into left tensor or vice versa
void HOTRG::contractionX(const int &D_cut, Tensor &leftT, Tensor &rightT, const double *U, const std::string mergeT) {
    assert(mergeT == "right" || mergeT == "left");
    const int Dx = leftT.GetDx(), Dy = leftT.GetDy();
    Tensor M(Dy * Dy, Dx);
    double sum;
    #pragma omp parallel private(sum)
    {
        #pragma omp for schedule(static)
        REP(i1, Dx)
            REP(j1, Dy)
                REP(l1, Dy)
                    REP(j2, Dy)
                        REP(k2, Dx)
                            REP(l2, Dy) {
                                sum = 0;
                                REP(a, Dx) sum += rightT(i1, j1, a, l1) * leftT(a, j2, k2, l2);
                                M(Dy * j1 + j2, i1, Dy * l1 + l2, k2) = sum;
                            }
    }
    MKL_INT Dy_new = std::min(Dy * Dy, D_cut);
    auto *temp = new double[Dy_new * Dy * Dy * Dx * Dx];
    #pragma omp parallel private(sum)
    {
        #pragma omp for schedule(static)
        REP(i, Dx)
            REP(k, Dx)
                REP(j, Dy_new)
                    REP(l, Dy * Dy) {
                        sum = 0;
                        REP(p, Dy * Dy) sum += U[Dy * Dy * p + j] * M(p, i, l, k);
                        temp[Dx * Dx * Dy * Dy * j + Dx * Dy * Dy * i + Dy * Dy * k + l] = sum;
                    }
    }
    #pragma omp parallel private(sum)
    {
        #pragma omp for schedule(static)
        REP(i, Dx)
            REP(k, Dx)
                REP(j, Dy_new)
                    REP(l, Dy_new) {
                        sum = 0;
                        REP(p, Dy * Dy) sum += U[Dy * Dy * p + l] *
                                               temp[Dx * Dx * Dy * Dy * j + Dx * Dy * Dy * i + Dy * Dy * k + p];
                        if (mergeT == "left") {
                            leftT(i, j, k, l) = sum;
                        } else {
                            rightT(i, j, k, l) = sum;
                        }
                    }
    }
    if (mergeT == "left") {
        leftT.UpdateDy(Dy_new);
    } else {
        rightT.UpdateDy(Dy_new);
    }
    delete [] temp;
}

// contraction top tensor into bottom tensor
void HOTRG::contractionY(const int &D_cut, Tensor &bottomT, Tensor &topT, const double *U, const std::string mergeT) {
    assert(mergeT == "bottom" || mergeT == "top");
    const int Dx = bottomT.GetDx(), Dy = bottomT.GetDy();
    Tensor M(Dx * Dx, Dy);
    double sum;
    #pragma omp parallel private(sum)
    {
        #pragma omp for schedule(static)
        REP(i1, Dx)
            REP(j1, Dy)
                REP(k1, Dx)
                    REP(i2, Dx)
                        REP(k2, Dx)
                            REP(l2, Dy) {
                                sum = 0;
                                REP(a, Dy) sum += topT(i1, j1, k1, a) * bottomT(i2, a, k2, l2);
                                M(Dx * i1 + i2, j1, Dx * k1 + k2, l2) = sum;
                            }
    }
    MKL_INT Dx_new = std::min(Dx * Dx, D_cut);
    auto *temp = new double[Dx_new * Dx * Dx * Dy * Dy];
    #pragma omp parallel private(sum)
    {
        #pragma omp for schedule(static)
        REP(j, Dy)
            REP(l, Dy)
                REP(i, Dx_new)
                    REP(k, Dx * Dx) {
                        sum = 0;
                        REP(p, Dx * Dx) sum += U[Dx * Dx * p + i] * M(p, j, k, l);
                        temp[Dx * Dx * Dy * Dy * i + Dy * Dy * k + Dy * j + l] = sum;
                    }
    }
    #pragma omp parallel private(sum)
    {
        #pragma omp for schedule(static)
        REP(j, Dy)
            REP(l, Dy)
                REP(i, Dx_new)
                    REP(k, Dx_new) {
                        sum = 0;
                        REP(p, Dx * Dx) sum += U[Dx * Dx * p + k] * temp[Dx * Dx * Dy * Dy * i + Dy * Dy * p + Dy * j + l];
                        if (mergeT == "top") {
                            topT(i, j, k, l) = sum;
                        } else {
                            bottomT(i, j, k, l) = sum;
                        }
                    }
    }
    if (mergeT == "top") {
        topT.UpdateDx(Dx_new);
    } else {
        bottomT.UpdateDx(Dx_new);
    }
    delete [] temp;
}