//
// Created by Joy on 2020/06/09.
//

#include "../include/HOTRG.hpp"
#include <mkl.h>
#include <iostream>
#include <iomanip>
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
    Tensor MM(Dx);
    Tensor A(Dx, Dy);
    Tensor B(Dx, Dy);
    double sum;
    /* compute Right Unitary matrix */
    REP(i, Dx)REP(i_, Dx)REP(p, Dy)REP(q, Dy) {
                    sum = 0;
                    REP(y, Dy)REP(k, Dx) {
                            sum += T(i, y, k, p) * T(i_, y, k, q);
                        }
                    A(i, p, i_, q) = sum;
                }
    REP(i, Dx)REP(i_, Dx)REP(p, Dy)REP(q, Dy) {
                    sum = 0;
                    REP(k, Dx)REP(y, Dy) {
                            sum += T(i, p, k, y) * T(i_, q, k, y);
                        }
                    B(i, p, i_, q) = sum;
                }
    REP4(i1, i1_, i2, i2_, Dx) {
                    sum = 0;
                    REP(p, Dy)REP(q, Dy) {
                            sum += A(i1, p, i1_, q) * B(i2, p, i2_, q);
                        }
                    MM(i1, i2, i1_, i2_) = sum;
                }
    const int m = Dx * Dx;
    auto temp = new double[m * m];
    auto sigma = new double[m];
    auto superb = new double[m - 1];
    MKL_INT info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'N', m, m, MM.GetMatrix(), m, sigma, U, m, nullptr, 1, superb);
    if (info > 0) {
        cerr << "The algorithm computing SVD failed to converge.\n";
        exit(1);
    }
    double epsilon_1 = 0, epsilon_2 = 0;
    for (int i = D_cut; i < m; ++i) {
        epsilon_1 += sigma[i];
    }
    if (epsilon_1 != 0) {
        /* compute Left Unitary matrix */
        REP(k1, Dx)REP(k1_, Dx)REP(p, Dy)REP(q, Dy) {
                        sum = 0;
                        REP(y, Dy)REP(i, Dx) {
                                sum += T(i, y, k1, p) * T(i, y, k1_, q);
                            }
                        A(k1, p, k1_, q) = sum;
                    }
        REP(k2, Dx)REP(k2_, Dx)REP(p, Dy)REP(q, Dy) {
                        sum = 0;
                        REP(i, Dx)REP(y, Dy) {
                                sum += T(i, p, k2, y) * T(i, q, k2_, y);
                            }
                        B(k2, p, k2_, q) = sum;
                    }
        REP4(k1, k1_, k2, k2_, Dx) {
                        sum = 0;
                        REP(p, Dy)REP(q, Dy) {
                                sum += A(k1, p, k1_, q) * B(k2, p, k2_, q);
                            }
                        MM(k1, k2, k1_, k2_) = sum;
                    }
        info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'N', m, m, MM.GetMatrix(), m, sigma, temp, m, nullptr, 1, superb);
        if (info > 0) {
            cerr << "The algorithm computing SVD failed to converge.\n";
            exit(1);
        }
        for (int i = D_cut; i < m; ++i) {
            epsilon_2 += sigma[i];
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
    double sum;
    Tensor MM(Dy);
    Tensor A(Dx, Dy);
    Tensor B(Dx, Dy);
    /* compute Up Unitary matrix */
    REP(j, Dy)REP(j_, Dy)REP(p, Dx)REP(q, Dx) {
                    sum = 0;
                    REP(l, Dy)REP(x, Dx) {
                            sum += T(x, j, p, l) * T(x, j_, q, l);
                        }
                    A(p, j, q, j_) = sum;
                }
    REP(j, Dy)REP(j_, Dy)REP(p, Dx)REP(q, Dx) {
                    sum = 0;
                    REP(l, Dy)REP(x, Dx) {
                            sum += T(p, j, x, l) * T(q, j_, x, l);
                        }
                    B(p, j, q, j_) = sum;
                }
    REP4(j1, j1_, j2, j2_, Dy) {
                    sum = 0;
                    REP(p, Dx)REP(q, Dx) {
                            sum += A(p, j1, q, j1_) * B(p, j2, q, j2_);
                        }
                    MM(j1, j2, j1_, j2_) = sum;
                }
    const int m = Dx * Dx;
    auto temp = new double [m * m];
    auto sigma = new double [m];
    auto superb = new double [m - 1];
    MKL_INT info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'N', m, m, MM.GetMatrix(), m, sigma, U, m, nullptr, 1, superb);
    if (info > 0) {
        cerr << "The algorithm computing SVD failed to converge.\n";
        exit(1);
    }
    double epsilon_1 = 0, epsilon_2 = 0;
    for (int i = D_cut; i < m; ++i) {
        epsilon_1 += sigma[i];
    }
    if (epsilon_1 != 0) {
        /* compute Down Unitary matrix */
        REP(l, Dy)REP(l_, Dy)REP(p, Dx)REP(q, Dx) {
                        sum = 0;
                        REP(j, Dy)REP(x, Dx) {
                                sum += T(x, j, p, l) * T(x, j, q, l_);
                            }
                        A(p, l, q, l_) = sum;
                    }
        REP(l, Dy)REP(l_, Dy)REP(p, Dx)REP(q, Dx) {
                        sum = 0;
                        REP(j, Dy)REP(x, Dx) {
                                sum += T(p, j, x, l) * T(q, j, x, l_);
                            }
                        B(p, l, q, l_) = sum;
                    }
        REP4(l1, l1_, l2, l2_, Dy) {
                        sum = 0;
                        REP(p, Dx)REP(q, Dx) {
                                sum += A(p, l1, q, l1_) * B(p, l2, q, l2_);
                            }
                        MM(l1, l2, l1_, l2_) = sum;
                    }
        info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'N', m, m, MM.GetMatrix(), m, sigma, temp, m, nullptr, 1, superb);
        if (info > 0) {
            cerr << "The algorithm computing SVD failed to converge.\n";
            exit(1);
        }
        for (int i = D_cut; i < m; ++i) {
            epsilon_2 += sigma[i];
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
    auto temp = new double[Dy_new * Dy * Dy * Dx * Dx];
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
    auto temp = new double[Dx_new * Dx * Dx * Dy * Dy];
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