#include "../include/HOTRG.hpp"
#include <mkl.h>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <vector>
#include <cmath>

#define REP(i, N) for (int i = 0; i < (N); ++i)
#define REP4(i, j, k, l, N) REP(i, N) REP(j, N) REP(k, N) REP(l, N)
#define REP4tensor(i, j, k, l, Dx, Dy) REP(i, Dx) REP(j, Dy) REP(k, Dx) REP(l, Dy)

using std::cout;
using std::cerr;

void HOTRG::Tensor::normalization(int c) {
    double _max = 0;
    this->forEach([&](int i, int j, int k, int l, const double *t) {
        const double absT = std::abs(*t);
        if (std::isnan(absT)) {
            std::cerr << "T(" << i << ',' << j << ',' << k << ',' << l << ") is nan";
            exit(1);
        }
        _max = std::max(_max, absT);
    });
    auto o = static_cast<int>(std::floor(std::log10(_max) / std::log10(c)));
    auto absO = std::abs(o);
    if (o > 0) {
        this->forEach([&](double *t) {
            REP(a, absO) *t /= c;
        });
    } else if (o < 0) {
        this->forEach([&](double *t) {
            REP(a, absO) *t *= c;
        });
    }
    orders.push_back(o);
}

void HOTRG::SVD_X(const int &D_cut, BaseTensor &T, double *U) {
    const int Dx = T.GetDx(), Dy = T.GetDy();
    BaseTensor MM(Dx);
    BaseTensor A(Dx, Dy, Dx, Dy);
    BaseTensor B(Dx, Dy, Dx, Dy);
    auto tmp1 = new double[Dx * Dx * Dy * Dy];
    auto tmp2 = new double[Dx * Dx * Dy * Dy];
    /* compute Right Unitary matrix */
    REP4tensor(i, j, k, l, Dx, Dy) {
                    tmp1[Dy * Dx * Dy * i + Dy * Dx * l + Dy * k + j] = T(i, j, k, l);
                    tmp2[Dy * Dx * Dy * k + Dy * Dx * j + Dy * i + l] = T(i, j, k, l);
                }
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Dx * Dy, Dx * Dy, Dx * Dy, 1, tmp1,
                Dx * Dy, tmp2, Dx * Dy, 0, A.GetMatrix(), Dx * Dy); // A(i, p, i_, q) = T(i, y, x, p) * T(i_, y, x, q)
    REP4tensor(i, j, k, l, Dx, Dy) {
                    tmp1[Dy * Dx * Dy * i + Dy * Dx * j + Dy * k + l] = T(i, j, k, l);
                    tmp2[Dy * Dx * Dy * k + Dy * Dx * l + Dy * i + j] = T(i, j, k, l);
                }
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Dx * Dy, Dx * Dy, Dx * Dy, 1, tmp1,
                Dx * Dy, tmp2, Dx * Dy, 0, B.GetMatrix(), Dx * Dy); // B(i, p, i_, q) = T(i, p, x, y) * T(i_, q, x, y)
    REP4tensor(i, j, k, l, Dx, Dy) {
                    tmp1[Dx * Dy * Dy * i + Dy * Dy * k + Dy * j + l] = A(i, j, k, l);
                    tmp2[Dy * Dx * Dx * j + Dx * Dx * l + Dx * i + k] = B(i, j, k, l);
                }
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Dx * Dx, Dx * Dx, Dy * Dy, 1, tmp1,
                Dy * Dy, tmp2, Dx * Dx, 0, MM.GetMatrix(), Dx * Dx);
    REP(i, Dx)REP(l, Dx) { // MM(i1, i2, i1_, i2_) = A(i1, p, i1_, q) * B(i2, p, i2_, q)
            auto t = new double[Dx * Dx];
            REP(j, Dx)REP(k, Dx) {
                    t[Dx * j + k] = MM(i, j, k, l);
                }
            REP(j, Dx)REP(k, Dx) {
                    MM(i, j, k, l) = t[Dx * k + j];
                }
            delete[] t;
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
        REP4tensor(i, j, k, l, Dx, Dy) {
                        tmp1[Dy * Dx * Dy * k + Dy * Dx * l + Dy * i + j] = T(i, j, k, l);
                        tmp2[Dy * Dx * Dy * i + Dy * Dx * j + Dy * k + l] = T(i, j, k, l);
                    }
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Dx * Dy, Dx * Dy, Dx * Dy, 1, tmp1,
                    Dx * Dy, tmp2, Dx * Dy, 0, A.GetMatrix(), Dx * Dy); // A(k, p, k_, q) = T(x, y, k, p) * T(x, y, k_, q)
        REP4tensor(i, j, k, l, Dx, Dy) {
                        tmp1[Dy * Dx * Dy * k + Dy * Dx * j + Dy * i + l] = T(i, j, k, l);
                        tmp2[Dy * Dx * Dy * i + Dy * Dx * l + Dy * k + j] = T(i, j, k, l);
                    }
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Dx * Dy, Dx * Dy, Dx * Dy, 1, tmp1,
                    Dx * Dy, tmp2, Dx * Dy, 0, B.GetMatrix(), Dx * Dy); // B(k, p, k_, q) = T(x, p, k, y) * T(x, q, k_, y)
        REP4tensor(i, j, k, l, Dx, Dy) {
                        tmp1[Dx * Dy * Dy * i + Dy * Dy * k + Dy * j + l] = A(i, j, k, l);
                        tmp2[Dy * Dx * Dx * j + Dx * Dx * l + Dx * i + k] = B(i, j, k, l);
                    }
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Dx * Dx, Dx * Dx, Dy * Dy, 1, tmp1,
                    Dy * Dy, tmp2, Dx * Dx, 0, MM.GetMatrix(), Dx * Dx);
        REP(i, Dx)REP(l, Dx) { // MM(k1, k2, k1_, k2_) = A(k1, p, k1_, q) * B(k2, p, k2_, q)
                auto t = new double[Dx * Dx];
                REP(j, Dx)REP(k, Dx) {
                        t[Dx * j + k] = MM(i, j, k, l);
                    }
                REP(j, Dx)REP(k, Dx) {
                        MM(i, j, k, l) = t[Dx * k + j];
                    }
                delete[] t;
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
    delete[] temp;
    delete[] sigma;
    delete[] superb;
    delete[] tmp1;
    delete[] tmp2;
}

void HOTRG::SVD_Y(const int &D_cut, BaseTensor &T, double *U) {
    const int Dx = T.GetDx(), Dy = T.GetDy();
    BaseTensor MM(Dy);
    BaseTensor A(Dx, Dy, Dx, Dy);
    BaseTensor B(Dx, Dy, Dx, Dy);
    auto tmp1 = new double[Dx * Dx * Dy * Dy];
    auto tmp2 = new double[Dx * Dx * Dy * Dy];
    /* compute Up Unitary matrix */
    REP4tensor(i, j, k, l, Dx, Dy) {
                    tmp1[Dy * Dx * Dy * k + Dy * Dx * j + Dy * i + l] = T(i, j, k, l);
                    tmp2[Dy * Dx * Dy * i + Dy * Dx * l + Dy * k + j] = T(i, j, k, l);
                }
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Dx * Dy, Dx * Dy, Dx * Dy, 1, tmp1,
                Dx * Dy, tmp2, Dx * Dy, 0, A.GetMatrix(), Dx * Dy); // A(p, j, q, j_) = T(x, j, p, y) * T(x, j_, q, y)
    REP4tensor(i, j, k, l, Dx, Dy) {
                    tmp1[Dy * Dx * Dy * i + Dy * Dx * j + Dy * k + l] = T(i, j, k, l);
                    tmp2[Dy * Dx * Dy * k + Dy * Dx * l + Dy * i + j] = T(i, j, k, l);
                }
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Dx * Dy, Dx * Dy, Dx * Dy, 1, tmp1,
                Dx * Dy, tmp2, Dx * Dy, 0, B.GetMatrix(), Dx * Dy); // B(p, j, q, j_) = T(p, j, x, y) * T(q, j_, x, y)
    REP4tensor(i, j, k, l, Dx, Dy) {
                    tmp1[Dy * Dx * Dx * j + Dx * Dx * l + Dx * i + k] = A(i, j, k, l);
                    tmp2[Dx * Dy * Dy * i + Dy * Dy * k + Dy * j + l] = B(i, j, k, l);
                }
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Dy * Dy, Dy * Dy, Dx * Dx, 1, tmp1,
                Dx * Dx, tmp2, Dy * Dy, 0, MM.GetMatrix(), Dy * Dy);
    REP(i, Dy)REP(l, Dy) { // MM(j1, j2, j1_, j2_) = A(p, j1, q, j1_) * B(p, j2, q, j2_)
            auto t = new double[Dy * Dy];
            REP(j, Dy)REP(k, Dy) {
                    t[Dy * j + k] = MM(i, j, k, l);
                }
            REP(j, Dy)REP(k, Dy) {
                    MM(i, j, k, l) = t[Dy * k + j];
                }
            delete[] t;
        }
    const int m = Dy * Dy;
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
        /* compute Down Unitary matrix */
        REP4tensor(i, j, k, l, Dx, Dy) {
                        tmp1[Dy * Dx * Dy * k + Dy * Dx * l + Dy * i + j] = T(i, j, k, l);
                        tmp2[Dy * Dx * Dy * i + Dy * Dx * j + Dy * k + l] = T(i, j, k, l);
                    }
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Dx * Dy, Dx * Dy, Dx * Dy, 1, tmp1,
                    Dx * Dy, tmp2, Dx * Dy, 0, A.GetMatrix(), Dx * Dy); // A(p, l, q, l_) = T(x, y, p, l) * T(x, y, q, l_)
        REP4tensor(i, j, k, l, Dx, Dy) {
                        tmp1[Dy * Dx * Dy * i + Dy * Dx * l + Dy * k + j] = T(i, j, k, l);
                        tmp2[Dy * Dx * Dy * k + Dy * Dx * j + Dy * i + l] = T(i, j, k, l);
                    }
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Dx * Dy, Dx * Dy, Dx * Dy, 1, tmp1,
                    Dx * Dy, tmp2, Dx * Dy, 0, B.GetMatrix(), Dx * Dy); // B(p, l, q, l_) = T(p, y, x, l) * T(q, y, x, l_)
        REP4tensor(i, j, k, l, Dx, Dy) {
                        tmp1[Dy * Dx * Dx * j + Dx * Dx * l + Dx * i + k] = A(i, j, k, l);
                        tmp2[Dx * Dy * Dy * i + Dy * Dy * k + Dy * j + l] = B(i, j, k, l);
                    }
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Dy * Dy, Dy * Dy, Dx * Dx, 1, tmp1,
                    Dx * Dx, tmp2, Dy * Dy, 0, MM.GetMatrix(), Dy * Dy);
        REP(i, Dy)REP(l, Dy) { // MM(l1, l2, l1_, l2_) = A(p, l1, q, l1_) * B(p, l2, q, l2_)
                auto t = new double[Dy * Dy];
                REP(j, Dy)REP(k, Dy) {
                        t[Dy * j + k] = MM(i, j, k, l);
                    }
                REP(j, Dy)REP(k, Dy) {
                        MM(i, j, k, l) = t[Dy * k + j];
                    }
                delete[] t;
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
    delete[] temp;
    delete[] sigma;
    delete[] superb;
    delete[] tmp1;
    delete[] tmp2;
}

// contraction right tensor into left tensor or vice versa
void HOTRG::contractionX(const int &D_cut, BaseTensor &leftT, BaseTensor &rightT, const double *U, const std::string mergeT) {
    assert(mergeT == "right" || mergeT == "left");
    const int Dx = leftT.GetDx(), Dy = leftT.GetDy(), Dy_new = std::min(Dy * Dy, D_cut);
    auto lT = new double[Dx * Dx * Dy * Dy];
    auto rT = new double[Dx * Dx * Dy * Dy];
    auto tU = new double[Dy_new * Dy * Dy];
    auto bU = new double[Dy_new * Dy * Dy];
    auto tmp1 = new double[Dy_new * Dx * Dx * Dy * Dy];
    auto tmp2 = new double[Dy_new * Dx * Dx * Dy * Dy];
    REP4tensor(i, j, k, l, Dx, Dy) {
                    lT[Dy * Dx * Dy * k + Dx * Dy * j + Dy * i + l] = leftT(i, j, k, l);
                    rT[Dx * Dy * Dy * i + Dy * Dy * k + Dy * l + j] = rightT(i, j, k, l);
                }
    REP(i, Dy)REP(j, Dy)REP(k, Dy_new) {
                tU[Dy_new * Dy * i + Dy_new * j + k] = U[Dy * Dy * Dy * i + Dy * Dy * j + k];
                bU[Dy_new * Dy * j + Dy_new * i + k] = U[Dy * Dy * Dy * i + Dy * Dy * j + k];
            }
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Dy * Dx * Dx, Dy * Dy_new, Dy, 1, rT,
                Dy, tU, Dy * Dy_new, 0, tmp1, Dy * Dy_new);
    delete[] rT;
    delete[] tU;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Dy * Dx * Dx, Dy * Dy_new, Dy, 1, lT,
                Dy, bU, Dy * Dy_new, 0, tmp2, Dy * Dy_new);
    delete[] lT;
    delete[] bU;
    auto tmp1_ = new double[Dy_new * Dx * Dx * Dy * Dy];
    REP(a, Dy)REP(b, Dx)REP(c, Dy)REP(i, Dx)REP(j, Dy_new) {
                        tmp1_[Dy_new * Dy * Dy * Dx * i + Dy * Dy * Dx * j + Dy * Dx * c + Dx * a + b]
                                = tmp1[Dx * Dy_new * Dy * Dy * i + Dy_new * Dy * Dy * b + Dy_new * Dy * c + Dy_new * a + j];
                    }
    delete[] tmp1;
    auto tmp2_ = new double[Dy_new * Dx * Dx * Dy * Dy];
    REP(a, Dy)REP(b, Dx)REP(c, Dy)REP(k, Dx)REP(l, Dy_new) {
                        tmp2_[Dy * Dx * Dx * Dy_new * c + Dx * Dx * Dy_new * a + Dx * Dy_new * b + Dy_new * k + l]
                                = tmp2[Dy * Dx * Dy * Dy_new * k + Dx * Dy * Dy_new * a + Dy * Dy_new * b + Dy_new * c + l];
                    }
    delete[] tmp2;
    if (mergeT == "left") {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Dx * Dy_new, Dx * Dy_new, Dx * Dy * Dy, 1,
                    tmp1_, Dx * Dy * Dy, tmp2_, Dx * Dy_new, 0, leftT.GetMatrix(), Dx * Dy_new);
        leftT.UpdateDy(Dy_new);
    } else {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Dx * Dy_new, Dx * Dy_new, Dx * Dy * Dy, 1,
                    tmp1_, Dx * Dy * Dy, tmp2_, Dx * Dy_new, 0, rightT.GetMatrix(), Dx * Dy_new);
        rightT.UpdateDy(Dy_new);
    }
    delete[] tmp1_;
    delete[] tmp2_;
}

// contraction top tensor into bottom tensor
void HOTRG::contractionY(const int &D_cut, BaseTensor &bottomT, BaseTensor &topT, const double *U, const std::string mergeT) {
    assert(mergeT == "bottom" || mergeT == "top");
    const int Dx = bottomT.GetDx(), Dy = bottomT.GetDy(), Dx_new = std::min(Dx * Dx, D_cut);
    auto bT = new double[Dx * Dx * Dy * Dy];
    auto tT = new double[Dx * Dx * Dy * Dy];
    auto lU = new double[Dx_new * Dx * Dx];
    auto rU = new double[Dx_new * Dx * Dx];
    auto tmp1 = new double[Dx_new * Dx * Dx * Dy * Dy];
    auto tmp2 = new double[Dx_new * Dx * Dx * Dy * Dy];
    REP4tensor(i, j, k, l, Dx, Dy) {
                    bT[Dy * Dy * Dx * i + Dy * Dx * j + Dx * l + k] = bottomT(i, j, k, l);
                    tT[Dx * Dy * Dx * j + Dy * Dx * k + Dx * l + i] = topT(i, j, k, l);
                }
    REP(i, Dx)REP(j, Dx)REP(k, Dx_new) {
                lU[Dx_new * Dx * j + Dx_new * i + k] = U[Dx * Dx * Dx * i + Dx * Dx * j + k];
                rU[Dx_new * Dx * i + Dx_new * j + k] = U[Dx * Dx * Dx * i + Dx * Dx * j + k];
            }
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Dy * Dy * Dx, Dx * Dx_new, Dx, 1, tT,
                Dx, rU, Dx * Dx_new, 0, tmp1, Dx * Dx_new);
    delete[] tT;
    delete[] rU;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Dy * Dy * Dx, Dx * Dx_new, Dx, 1, bT,
                Dx, lU, Dx * Dx_new, 0, tmp2, Dx * Dx_new);
    delete[] bT;
    delete[] lU;
    auto tmp1_ = new double[Dx_new * Dx * Dx * Dy * Dy];
    REP(a, Dx)REP(b, Dy)REP(c, Dx)REP(j, Dy)REP(i, Dx_new) {
                        tmp1_[Dx * Dy * Dx * Dy * i + Dx * Dy * Dx * j + Dx * Dy * a + Dx * b + c]
                                = tmp1[Dx_new * Dx * Dy * Dx * j + Dx_new * Dx * Dy * a + Dx_new * Dx * b + Dx_new * c + i];
                    }
    delete[] tmp1;
    auto tmp2_ = new double[Dx_new * Dx * Dx * Dy * Dy];
    REP(a, Dx)REP(b, Dy)REP(c, Dx)REP(l, Dy)REP(k, Dx_new) {
                        tmp2_[Dy * Dx_new * Dx * Dy * a + Dy * Dx_new * Dx * b + Dy * Dx_new * c + Dy * k + l]
                                = tmp2[Dx_new * Dx * Dy * Dy * c + Dx_new * Dx * Dy * b + Dx_new * Dx * l + Dx_new * a + k];
                    }
    delete[] tmp2;
    if (mergeT == "top") {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Dy * Dx_new, Dy * Dx_new, Dx * Dx * Dy, 1,
                    tmp1_, Dx * Dx * Dy, tmp2_, Dy * Dx_new, 0, topT.GetMatrix(), Dy * Dx_new);
        topT.UpdateDx(Dx_new);
    } else {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Dy * Dx_new, Dy * Dx_new, Dx * Dx * Dy, 1,
                    tmp1_, Dx * Dx * Dy, tmp2_, Dy * Dx_new, 0, bottomT.GetMatrix(), Dy * Dx_new);
        bottomT.UpdateDx(Dx_new);
    }
    delete[] tmp1_;
    delete[] tmp2_;
}