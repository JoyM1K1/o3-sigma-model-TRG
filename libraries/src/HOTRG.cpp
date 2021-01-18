#include "../include/HOTRG.hpp"
#include "../include/spherical_harmonics.hpp"
#include "../include/gauss_quadrature.hpp"
#include "../include/time_counter.hpp"
#include <mkl.h>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <vector>
#include <cmath>

#define REP(i, N) for (int i = 0; i < (N); ++i)
#define REP4tensor(i, j, k, l, Dx, Dy) REP(i, Dx) REP(j, Dy) REP(k, Dx) REP(l, Dy)

using std::cout;
using std::cerr;

void HOTRG::initialize_spherical_harmonics(Tensor &T, const double &beta, const int &D_cut, const int &l_max) {
    time_counter time;
    time.start();
    cout << "initialize tensor " << std::flush;
    T = Tensor(D_cut);
    SphericalHarmonics::init_tensor(beta, l_max, T);
    time.end();
    cout << "in " << time.duration_cast_to_string() << "\n" << std::flush;
}

void HOTRG::initialize_gauss_quadrature(Tensor &T, const double &beta, const int &D_cut, const int &n_node) {
    time_counter time;
    time.start();
    cout << "initialize tensor " << std::flush;
    const int D = std::min(D_cut, n_node * n_node);
    T = Tensor(D, D_cut);
    GaussQuadrature::init_tensor(beta, n_node, D_cut, T);
    time.end();
    cout << "in " << time.duration_cast_to_string() << "\n" << std::flush;
}

void HOTRG::initialize_spherical_harmonics_with_impure(Tensor &T, ImpureTensor &IMT, const double &beta, const int &D_cut, const int &l_max) {
    time_counter time;
    time.start();
    cout << "initialize tensor " << std::flush;
    T = Tensor(D_cut);
    IMT = ImpureTensor(D_cut);
    SphericalHarmonics::init_tensor_with_impure(beta, l_max, T, IMT);
    time.end();
    cout << "in " << time.duration_cast_to_string() << '\n' << std::flush;
}

void HOTRG::initialize_gauss_quadrature_with_impure(Tensor &T, ImpureTensor &IMT, const double &beta, const int &D_cut, const int &n_node) {
    time_counter time;
    time.start();
    cout << "initialize tensor " << std::flush;
    const int D = std::min(D_cut, n_node * n_node);
    T = Tensor(D, D_cut);
    IMT = ImpureTensor(D, D_cut);
    GaussQuadrature::init_tensor_with_impure(beta, n_node, D_cut, D, T, IMT);
    time.end();
    cout << "in " << time.duration_cast_to_string() << '\n' << std::flush;
}

void HOTRG::SVD_X(const int &D_cut, BaseTensor &T, double *&U) {
    const int Dx = T.GetDx(), Dy = T.GetDy(), D_max = T.GetD_max();
    BaseTensor MM(Dx, D_max);
    BaseTensor MM_(Dx, D_max);
    BaseTensor A(Dx, Dy, Dx, Dy);
    BaseTensor B(Dx, Dy, Dx, Dy);
    BaseTensor tmp_1(Dx, Dy, Dx, Dy, D_max);
    BaseTensor tmp_2(Dx, Dy, Dx, Dy, D_max);
    /* compute Right Unitary matrix */
    T.forEach([&](int i, int j, int k, int l, const double *t) {
        tmp_1(i, l, k, j) = *t;
        tmp_2(k, j, i, l) = *t;
    });
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Dx * Dy, Dx * Dy, Dx * Dy, 1, tmp_1.GetMatrix(),
            Dx * Dy, tmp_2.GetMatrix(), Dx * Dy, 0, A.GetMatrix(), Dx * Dy); // A(i, p, i_, q) = T(i, y, x, p) * T(i_, y, x, q)
    T.forEach([&](int i, int j, int k, int l, const double *t) {
        tmp_1(i, j, k, l) = *t;
        tmp_2(k, l, i, j) = *t;
    });
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Dx * Dy, Dx * Dy, Dx * Dy, 1, tmp_1.GetMatrix(),
            Dx * Dy, tmp_2.GetMatrix(), Dx * Dy, 0, B.GetMatrix(), Dx * Dy); // B(i, p, i_, q) = T(i, p, x, y) * T(i_, q, x, y)
    tmp_1.SetDj(Dx); tmp_1.SetDk(Dy); // (Dx, Dy, Dx, Dy) -> (Dx, Dx, Dy, Dy)
    tmp_2.SetDi(Dy); tmp_2.SetDl(Dx); // (Dx, Dy, Dx, Dy) -> (Dy, Dy, Dx, Dx)
    A.forEach([&](int i, int j, int k, int l, const double *t) {
        tmp_1(i, k, j, l) = *t;
    });
    B.forEach([&](int i, int j, int k, int l, const double *t) {
        tmp_2(j, l, i, k) = *t;
    });
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Dx * Dx, Dx * Dx, Dy * Dy, 1, tmp_1.GetMatrix(),
            Dy * Dy, tmp_2.GetMatrix(), Dx * Dx, 0, MM.GetMatrix(), Dx * Dx);
    MM.forEach([&](int i, int j, int k, int l, const double *t) {
        MM_(i, k, j, l) = *t;
    }); // MM(i1, i2, i1_, i2_) = A(i1, p, i1_, q) * B(i2, p, i2_, q)
    const int m = Dx * Dx;
    auto sigma = new double[m];
    auto superb = new double[m - 1];
    MKL_INT info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'N', m, m, MM_.GetMatrix(), m, sigma, U, m, nullptr, 1, superb);
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
        tmp_1.SetDj(Dy); tmp_1.SetDk(Dx); // (Dx, Dx, Dy, Dy) -> (Dx, Dy, Dx, Dy)
        tmp_2.SetDi(Dx); tmp_2.SetDl(Dy); // (Dy, Dy, Dx, Dx) -> (Dx, Dy, Dx, Dy)
        T.forEach([&](int i, int j, int k, int l, const double *t) {
            tmp_1(k, l, i, j) = *t;
            tmp_2(i, j, k, l) = *t;
        });
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Dx * Dy, Dx * Dy, Dx * Dy, 1, tmp_1.GetMatrix(),
                Dx * Dy, tmp_2.GetMatrix(), Dx * Dy, 0, A.GetMatrix(), Dx * Dy); // A(k, p, k_, q) = T(x, y, k, p) * T(x, y, k_, q)
        T.forEach([&](int i, int j, int k, int l, const double *t) {
            tmp_1(k, j, i, l) = *t;
            tmp_2(i, l, k, j) = *t;
        });
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Dx * Dy, Dx * Dy, Dx * Dy, 1, tmp_1.GetMatrix(),
                Dx * Dy, tmp_2.GetMatrix(), Dx * Dy, 0, B.GetMatrix(), Dx * Dy); // B(k, p, k_, q) = T(x, p, k, y) * T(x, q, k_, y)
        tmp_1.SetDj(Dx); tmp_1.SetDk(Dy); // (Dx, Dy, Dx, Dy) -> (Dx, Dx, Dy, Dy)
        tmp_2.SetDi(Dy); tmp_2.SetDl(Dx); // (Dx, Dy, Dx, Dy) -> (Dy, Dy, Dx, Dx)
        A.forEach([&](int i, int j, int k, int l, const double *t) {
            tmp_1(i, k, j, l) = *t;
        });
        B.forEach([&](int i, int j, int k, int l, const double *t) {
            tmp_2(j, l, i, k) = *t;
        });
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Dx * Dx, Dx * Dx, Dy * Dy, 1, tmp_1.GetMatrix(),
                Dy * Dy, tmp_2.GetMatrix(), Dx * Dx, 0, MM.GetMatrix(), Dx * Dx);
        MM.forEach([&](int i, int j, int k, int l, const double *t) {
            MM_(i, k, j, l) = *t;
        }); // MM(k1, k2, k1_, k2_) = A(k1, p, k1_, q) * B(k2, p, k2_, q)
        auto tmpU = new double[m * m];
        info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'N', m, m, MM_.GetMatrix(), m, sigma, tmpU, m, nullptr, 1, superb);
        if (info > 0) {
            cerr << "The algorithm computing SVD failed to converge.\n";
            exit(1);
        }
        for (int i = D_cut; i < m; ++i) {
            epsilon_2 += sigma[i];
        }
        if (epsilon_1 > epsilon_2) {
            delete[] U;
            U = tmpU;
        } else {
            delete[] tmpU;
        }
    }
    delete[] sigma;
    delete[] superb;
}

void HOTRG::SVD_Y(const int &D_cut, BaseTensor &T, double *&U) {
    const int Dx = T.GetDx(), Dy = T.GetDy(), D_max = T.GetD_max();
    BaseTensor MM(Dy, D_max);
    BaseTensor MM_(Dy, D_max);
    BaseTensor A(Dx, Dy, Dx, Dy);
    BaseTensor B(Dx, Dy, Dx, Dy);
    BaseTensor tmp_1(Dx, Dy, Dx, Dy, D_max);
    BaseTensor tmp_2(Dx, Dy, Dx, Dy, D_max);
    /* compute Up Unitary matrix */
    T.forEach([&](int i, int j, int k, int l, const double *t) {
        tmp_1(k, j, i, l) = *t;
        tmp_2(i, l, k, j) = *t;
    });
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Dx * Dy, Dx * Dy, Dx * Dy, 1, tmp_1.GetMatrix(),
            Dx * Dy, tmp_2.GetMatrix(), Dx * Dy, 0, A.GetMatrix(), Dx * Dy); // A(p, j, q, j_) = T(x, j, p, y) * T(x, j_, q, y)
    T.forEach([&](int i, int j, int k, int l, const double *t) {
        tmp_1(i, j, k, l) = *t;
        tmp_2(k, l, i, j) = *t;
    });
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Dx * Dy, Dx * Dy, Dx * Dy, 1, tmp_1.GetMatrix(),
            Dx * Dy, tmp_2.GetMatrix(), Dx * Dy, 0, B.GetMatrix(), Dx * Dy); // B(p, j, q, j_) = T(p, j, x, y) * T(q, j_, x, y)
    tmp_1.SetDi(Dy); tmp_1.SetDl(Dx); // (Dx, Dy, Dx, Dy) -> (Dy, Dy, Dx, Dx)
    tmp_2.SetDj(Dx); tmp_2.SetDk(Dy); // (Dx, Dy, Dx, Dy) -> (Dx, Dx, Dy, Dy)
    A.forEach([&](int i, int j, int k, int l, const double *t) {
        tmp_1(j, l, i, k) = *t;
    });
    B.forEach([&](int i, int j, int k, int l, const double *t) {
        tmp_2(i, k, j, l) = *t;
    });
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Dy * Dy, Dy * Dy, Dx * Dx, 1, tmp_1.GetMatrix(),
            Dx * Dx, tmp_2.GetMatrix(), Dy * Dy, 0, MM.GetMatrix(), Dy * Dy);
    MM.forEach([&](int i, int j, int k, int l, const double *t) {
        MM_(i, k, j, l) = *t;
    });
    const int m = Dy * Dy;
    auto sigma = new double[m];
    auto superb = new double[m - 1];
    MKL_INT info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'N', m, m, MM_.GetMatrix(), m, sigma, U, m, nullptr, 1, superb);
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
        tmp_1.SetDi(Dx); tmp_1.SetDl(Dy); // (Dy, Dy, Dx, Dx) -> (Dx, Dy, Dx, Dy)
        tmp_2.SetDj(Dy); tmp_2.SetDk(Dx); // (Dx, Dx, Dy, Dy) -> (Dx, Dy, Dx, Dy)
        T.forEach([&](int i, int j, int k, int l, const double *t) {
            tmp_1(k, l, i, j) = *t;
            tmp_2(i, j, k, l) = *t;
        });
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Dx * Dy, Dx * Dy, Dx * Dy, 1, tmp_1.GetMatrix(),
                Dx * Dy, tmp_2.GetMatrix(), Dx * Dy, 0, A.GetMatrix(), Dx * Dy); // A(p, l, q, l_) = T(x, y, p, l) * T(x, y, q, l_)
        T.forEach([&](int i, int j, int k, int l, const double *t) {
            tmp_1(i, l, k, j) = *t;
            tmp_2(k, j, i, l) = *t;
        });
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Dx * Dy, Dx * Dy, Dx * Dy, 1, tmp_1.GetMatrix(),
                Dx * Dy, tmp_2.GetMatrix(), Dx * Dy, 0, B.GetMatrix(), Dx * Dy); // B(p, l, q, l_) = T(p, y, x, l) * T(q, y, x, l_)
        tmp_1.SetDi(Dy); tmp_1.SetDl(Dx); // (Dx, Dy, Dx, Dy) -> (Dy, Dy, Dx, Dx)
        tmp_2.SetDj(Dx); tmp_2.SetDk(Dy); // (Dx, Dy, Dx, Dy) -> (Dx, Dx, Dy, Dy)
        A.forEach([&](int i, int j, int k, int l, const double *t) {
            tmp_1(j, l, i, k) = *t;
        });
        B.forEach([&](int i, int j, int k, int l, const double *t) {
            tmp_2(i, k, j, l) = *t;
        });
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Dy * Dy, Dy * Dy, Dx * Dx, 1, tmp_1.GetMatrix(),
                Dx * Dx, tmp_2.GetMatrix(), Dy * Dy, 0, MM.GetMatrix(), Dy * Dy);
        MM.forEach([&](int i, int j, int k, int l, const double *t) {
            MM_(i, k, j, l) = *t;
        });
        auto tmpU = new double[m * m];
        info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'N', m, m, MM_.GetMatrix(), m, sigma, tmpU, m, nullptr, 1, superb);
        if (info > 0) {
            cerr << "The algorithm computing SVD failed to converge.\n";
            exit(1);
        }
        for (int i = D_cut; i < m; ++i) {
            epsilon_2 += sigma[i];
        }
        if (epsilon_1 > epsilon_2) {
            delete[] U;
            U = tmpU;
        } else {
            delete[] tmpU;
        }
    }
    delete[] sigma;
    delete[] superb;
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

double HOTRG::renormalization::partition_alt(Tensor &T, long long int *orders, const int &n, const int &normalize_factor) {
    const int D_cut = T.GetD_max();

    /* normalization */
    orders[n - 1] = T.normalization(normalize_factor);

    if (n % 2) { // compression along x-axis
        const int Dy = T.GetDy();
        auto U = new double[Dy * Dy * Dy * Dy];
        HOTRG::SVD_Y(D_cut, T, U);
        HOTRG::contractionX(D_cut, T, T, U, "left");
        delete[] U;
    } else { // compression along y-axis
        const int Dx = T.GetDx();
        auto U = new double[Dx * Dx * Dx * Dx];
        HOTRG::SVD_X(D_cut, T, U);
        HOTRG::contractionY(D_cut, T, T, U, "bottom");
        delete[] U;
    }

    double Tr = T.trace();
    Tr = std::log(Tr);
    REP(i, n) Tr /= 2; // 体積で割る
    REP(i, n) {
        double tmp = orders[i] * std::log(normalize_factor);
        REP(j, i) tmp /= 2;
        Tr += tmp;
    }
    return Tr;
}

void HOTRG::renormalization::one_point_alt(Tensor &T, ImpureTensor &IMT, long long *orders, const int &n, const int &normalize_factor, double *res) {
    const int D_cut = T.GetD_max();
    if (n % 2) { // compress along x-axis
        cout << " compress along x-axis : " << std::flush;
        const int Dy = T.GetDy();
        auto U = new double[Dy * Dy * Dy * Dy];
        HOTRG::SVD_Y(D_cut, T, U);
        for (auto &tensor : IMT.tensors) HOTRG::contractionX(D_cut, tensor, T, U, "left");
        HOTRG::contractionX(D_cut, T, T, U, "left");
        delete[] U;
    } else { // compress along y-axis
        cout << " compress along y-axis : " << std::flush;
        const int Dx = T.GetDx();
        auto U = new double[Dx * Dx * Dx * Dx];
        HOTRG::SVD_X(D_cut, T, U);
        for (auto &tensor : IMT.tensors) HOTRG::contractionY(D_cut, tensor, T, U, "bottom");
        HOTRG::contractionY(D_cut, T, T, U, "bottom");
        delete[] U;
    }

    /* normalization */
    T.normalization(normalize_factor);
    for (auto &tensor : IMT.tensors) tensor.normalization(normalize_factor);
    REP(i, DIMENSION) orders[i] += IMT.tensors[i].order - T.order;

    double Tr = T.trace();
    REP(i, DIMENSION) {
        double impureTr = IMT.tensors[i].trace();
        const long long int order = orders[i];
        const long long int absOrder = std::abs(order);
        if (order > 0) {
            REP(k, absOrder) impureTr *= normalize_factor;
        } else {
            REP(k, absOrder) impureTr /= normalize_factor;
        }
        res[i] = impureTr / Tr;
    }
}

void
HOTRG::renormalization::two_point_alt(Tensor &T, ImpureTensor &IMT, long long *orders, const int &n, const int &merge_point, const int &normalize_factor, double *res) {
    const int D_cut = T.GetD_max();
    const int times = (n + 1) / 2;
    if (n % 2) { // compress along x-axis
        cout << " compress along x-axis " << std::flush;
        const int Dy = T.GetDy();
        auto U = new double[Dy * Dy * Dy * Dy];
        HOTRG::SVD_Y(D_cut, T, U);
        if (times == merge_point) {
            for (auto &tensor : IMT.tensors) {
                HOTRG::contractionX(D_cut, tensor, tensor, U, "left");
            }
        } else {
            for (auto &tensor : IMT.tensors) {
                HOTRG::contractionX(D_cut, tensor, T, U, "left");
            }
        }
        HOTRG::contractionX(D_cut, T, T, U, "left");
        delete[] U;
    } else { // compress along y-axis
        cout << " compress along y-axis " << std::flush;
        const int Dx = T.GetDx();
        auto U = new double[Dx * Dx * Dx * Dx];
        HOTRG::SVD_X(D_cut, T, U);
        for (auto &tensor : IMT.tensors) HOTRG::contractionY(D_cut, tensor, T, U, "bottom");
        HOTRG::contractionY(D_cut, T, T, U, "bottom");
        delete[] U;
    }

    /* normalization */
    T.normalization(normalize_factor);
    for (auto &tensor : IMT.tensors) tensor.normalization(normalize_factor);
    REP(i, DIMENSION) {
        long long int order = IMT.tensors[i].order - T.order;
        if (times < merge_point) {
            order *= 2;
        }
        orders[i] += order;
    }

    double Tr = T.trace();

    REP(i, DIMENSION) {
        double impureTr = IMT.tensors[i].trace();
        const long long int order = orders[i];
        const long long int absOrder = std::abs(order);
        if (order > 0) {
            REP(k, absOrder) impureTr *= normalize_factor;
        } else {
            REP(k, absOrder) impureTr /= normalize_factor;
        }
        res[i] = impureTr / Tr;
    }
}

void
HOTRG::renormalization::two_point_manual(Tensor &T, ImpureTensor &originIMT, ImpureTensor &IMT, long long *orders, const int &n, std::pair<int, int> &p, const int &normalize_factor, double *res) {
    const int D_cut = T.GetD_max();
    if (n % 2) { // compression along x-axis
        cout << " compress along x-axis " << std::flush;
        cout << std::setw(7);
        const int Dy = T.GetDy();
        auto U = new double[Dy * Dy * Dy * Dy];
        HOTRG::SVD_Y(D_cut, T, U);
        if (p.first) {
            if (p.first == 1) {
                if (p.second) {
                    cout << "right" << std::flush;
                    for (auto &tensor : IMT.tensors) {
                        HOTRG::contractionX(D_cut, T, tensor, U, "right");
                    }
                } else {
                    cout << "merged" << std::flush;
                    REP(i, DIMENSION) {
                        HOTRG::contractionX(D_cut, originIMT.tensors[i], IMT.tensors[i], U, "right");
                    }
                    IMT.isMerged = true;
                }
            } else if (p.first & 1) {
                cout << "right" << std::flush;
                for (auto &tensor : IMT.tensors) {
                    HOTRG::contractionX(D_cut, T, tensor, U, "right");
                }
            } else {
                cout << "left" << std::flush;
                for (auto &tensor : IMT.tensors) {
                    HOTRG::contractionX(D_cut, tensor, T, U, "left");
                }
            }
            p.first >>= 1;
        } else {
            cout << "left" << std::flush;
            for (auto &tensor : IMT.tensors) {
                HOTRG::contractionX(D_cut, tensor, T, U, "left");
            }
        }
        if (!IMT.isMerged) {
            for (auto &tensor : originIMT.tensors) {
                HOTRG::contractionX(D_cut, tensor, T, U, "left");
            }
        }
        HOTRG::contractionX(D_cut, T, T, U, "left");
        delete[] U;
    } else { // compression along y-axis
        cout << " compress along y-axis " << std::flush;
        cout << std::setw(7);
        const int Dx = T.GetDx();
        auto U = new double[Dx * Dx * Dx * Dx];
        HOTRG::SVD_X(D_cut, T, U);
        if (p.second) {
            if (p.second == 1) {
                if (p.first) {
                    cout << "top" << std::flush;
                    for (auto &tensor : IMT.tensors) {
                        HOTRG::contractionY(D_cut, T, tensor, U, "top");
                    }
                } else {
                    cout << "merged" << std::flush;
                    REP(i, DIMENSION) {
                        HOTRG::contractionY(D_cut, originIMT.tensors[i], IMT.tensors[i], U, "top");
                    }
                    IMT.isMerged = true;
                }
            } else if (p.second & 1) {
                cout << "top" << std::flush;
                for (auto &tensor : IMT.tensors) {
                    HOTRG::contractionY(D_cut, T, tensor, U, "top");
                }
            } else {
                cout << "bottom" << std::flush;
                for (auto &tensor : IMT.tensors) {
                    HOTRG::contractionY(D_cut, tensor, T, U, "bottom");
                }
            }
            p.second >>= 1;
        } else {
            cout << "bottom" << std::flush;
            for (auto &tensor : IMT.tensors) {
                HOTRG::contractionY(D_cut, tensor, T, U, "bottom");
            }
        }
        if (!IMT.isMerged) {
            for (auto &tensor : originIMT.tensors) {
                HOTRG::contractionY(D_cut, tensor, T, U, "bottom");
            }
        }
        HOTRG::contractionY(D_cut, T, T, U, "bottom");
        delete[] U;
    }

    /* normalization */
    T.normalization(normalize_factor);
    if (p.first || p.second) {
        REP(i, DIMENSION) {
            originIMT.tensors[i].normalization(normalize_factor);
            orders[i] += originIMT.tensors[i].order - T.order;
        }
    }
    REP(i, DIMENSION) {
        IMT.tensors[i].normalization(normalize_factor);
        orders[i] += IMT.tensors[i].order - T.order;
    }

    double Tr = T.trace();

    REP(k, DIMENSION) {
        long long int order = orders[k];
        double impureTr = IMT.tensors[k].trace();
        unsigned long long int absOrder = std::abs(order);
        if (order > 0) {
            REP(i, absOrder) impureTr *= normalize_factor;
        } else {
            REP(i, absOrder) impureTr /= normalize_factor;
        }
        res[k] = impureTr / Tr;
    }
}

void
HOTRG::renormalization::mass(Tensor &T, ImpureTensor &IMT, long long *orders, const int &N, const int &n, const int &merge_point, const int &normalize_factor, double *res) {
    const int D_cut = T.GetD_max();
    const int times = n - N / 2;
    if (n > N / 2) { // compress along x-axis
        cout << " compress along x-axis " << std::flush;
        const int Dy = T.GetDy();
        auto U = new double[Dy * Dy * Dy * Dy];
        HOTRG::SVD_Y(D_cut, T, U);
        if (times == merge_point) {
            for (auto &tensor : IMT.tensors) {
                HOTRG::contractionX(D_cut, tensor, tensor, U, "left");
            }
        } else {
            for (auto &tensor : IMT.tensors) {
                HOTRG::contractionX(D_cut, tensor, T, U, "left");
            }
        }
        HOTRG::contractionX(D_cut, T, T, U, "left");
        delete[] U;
    } else { // compress along y-axis
        cout << " compress along y-axis " << std::flush;
        const int Dx = T.GetDx();
        auto U = new double[Dx * Dx * Dx * Dx];
        HOTRG::SVD_X(D_cut, T, U);
        auto imt1 = IMT;
        auto imt2 = IMT;
        for (int a = 0; a < 3; ++a) {
            HOTRG::contractionY(D_cut, imt1.tensors[a], T, U, "bottom");
            HOTRG::contractionY(D_cut, T, imt2.tensors[a], U, "top");
            IMT.tensors[a].UpdateDx(imt1.tensors[a].GetDx());
            IMT.tensors[a].forEach([&](int i, int j, int k, int l, double *t) {
                *t = imt1.tensors[a](i, j, k, l) + imt2.tensors[a](i, j, k, l);
            });
        }
        HOTRG::contractionY(D_cut, T, T, U, "bottom");
        delete[] U;
    }

    /* normalization */
    T.normalization(normalize_factor);
    for (auto &tensor : IMT.tensors) tensor.normalization(normalize_factor);
    REP(i, DIMENSION) {
        long long int order = IMT.tensors[i].order - T.order;
        if (times < merge_point) {
            order *= 2;
        }
        orders[i] += order;
    }

    double Tr = T.trace();

    REP(i, DIMENSION) {
        double impureTr = IMT.tensors[i].trace();
        const long long int order = orders[i];
        const long long int absOrder = std::abs(order);
        if (order > 0) {
            REP(k, absOrder) impureTr *= normalize_factor;
        } else {
            REP(k, absOrder) impureTr /= normalize_factor;
        }
        res[i] = impureTr / Tr;
    }
}

void
HOTRG::renormalization::mass_alt(Tensor &T, ImpureTensor &IMT, long long *orders, const int &n, const int &merge_point, const int &normalize_factor, double *res) {
    const int D_cut = T.GetD_max();
    const int times = (n + 1) / 2;
    if (n % 2) { // compress along x-axis
        cout << " compress along x-axis " << std::flush;
        const int Dy = T.GetDy();
        auto U = new double[Dy * Dy * Dy * Dy];
        HOTRG::SVD_Y(D_cut, T, U);
        if (times == merge_point) {
            for (auto &tensor : IMT.tensors) {
                HOTRG::contractionX(D_cut, tensor, tensor, U, "left");
            }
        } else {
            for (auto &tensor : IMT.tensors) {
                HOTRG::contractionX(D_cut, tensor, T, U, "left");
            }
        }
        HOTRG::contractionX(D_cut, T, T, U, "left");
        delete[] U;
    } else { // compress along y-axis
        cout << " compress along y-axis " << std::flush;
        const int Dx = T.GetDx();
        auto U = new double[Dx * Dx * Dx * Dx];
        HOTRG::SVD_X(D_cut, T, U);
        auto imt1 = IMT;
        auto imt2 = IMT;
        for (int a = 0; a < 3; ++a) {
            HOTRG::contractionY(D_cut, imt1.tensors[a], T, U, "bottom");
            HOTRG::contractionY(D_cut, T, imt2.tensors[a], U, "top");
            IMT.tensors[a].UpdateDx(imt1.tensors[a].GetDx());
            IMT.tensors[a].forEach([&](int i, int j, int k, int l, double *t) {
                *t = imt1.tensors[a](i, j, k, l) + imt2.tensors[a](i, j, k, l);
            });
        }
        HOTRG::contractionY(D_cut, T, T, U, "bottom");
        delete[] U;
    }

    /* normalization */
    T.normalization(normalize_factor);
    for (auto &tensor : IMT.tensors) tensor.normalization(normalize_factor);
    REP(i, DIMENSION) {
        long long int order = IMT.tensors[i].order - T.order;
        if (times < merge_point) {
            order *= 2;
        }
        orders[i] += order;
    }

    double Tr = T.trace();

    REP(i, DIMENSION) {
        double impureTr = IMT.tensors[i].trace();
        const long long int order = orders[i];
        const long long int absOrder = std::abs(order);
        if (order > 0) {
            REP(k, absOrder) impureTr *= normalize_factor;
        } else {
            REP(k, absOrder) impureTr /= normalize_factor;
        }
        res[i] = impureTr / Tr;
    }
}

void
HOTRG::renormalization::mass_manual(Tensor &T, ImpureTensor &originIMT, ImpureTensor &IMT, long long *orders, const int &N, const int &n, int &distance, const int &normalize_factor, double *res) {
    const int D_cut = T.GetD_max();
    if (n > N / 2) { // compress along x-axis
        cout << " compress along x-axis " << std::flush;
        cout << std::setw(7);
        const int Dy = T.GetDy();
        auto U = new double[Dy * Dy * Dy * Dy];
        HOTRG::SVD_Y(D_cut, T, U);
        if (distance) {
            if (distance == 1) {
                cout << "merged" << std::flush;
                REP(i, DIMENSION) {
                    HOTRG::contractionX(D_cut, originIMT.tensors[i], IMT.tensors[i], U, "right");
                }
                IMT.isMerged = true;
            } else if (distance & 1) {
                cout << "right" << std::flush;
                for (auto &tensor : IMT.tensors) {
                    HOTRG::contractionX(D_cut, T, tensor, U, "right");
                }
            } else {
                cout << "left" << std::flush;
                for (auto &tensor : IMT.tensors) {
                    HOTRG::contractionX(D_cut, tensor, T, U, "left");
                }
            }
            distance >>= 1;
        } else {
            cout << "left" << std::flush;
            for (auto &tensor : IMT.tensors) {
                HOTRG::contractionX(D_cut, tensor, T, U, "left");
            }
        }
        if (!IMT.isMerged) {
            for (auto &tensor : originIMT.tensors) {
                HOTRG::contractionX(D_cut, tensor, T, U, "left");
            }
        }
        HOTRG::contractionX(D_cut, T, T, U, "left");
        delete[] U;
    } else { // compress along y-axis
        cout << " compress along y-axis " << std::flush;
        const int Dx = T.GetDx();
        auto U = new double[Dx * Dx * Dx * Dx];
        HOTRG::SVD_X(D_cut, T, U);
        auto imt1 = IMT;
        auto imt2 = IMT;
        for (int a = 0; a < 3; ++a) {
            HOTRG::contractionY(D_cut, imt1.tensors[a], T, U, "bottom");
            HOTRG::contractionY(D_cut, T, imt2.tensors[a], U, "top");
            IMT.tensors[a].UpdateDx(imt1.tensors[a].GetDx());
            IMT.tensors[a].forEach([&](int i, int j, int k, int l, double *t) {
                *t = imt1.tensors[a](i, j, k, l) + imt2.tensors[a](i, j, k, l);
            });
        }
        if (!IMT.isMerged) {
            imt1 = originIMT;
            imt2 = originIMT;
            for (int a = 0; a < 3; ++a) {
                HOTRG::contractionY(D_cut, imt1.tensors[a], T, U, "bottom");
                HOTRG::contractionY(D_cut, T, imt2.tensors[a], U, "top");
                originIMT.tensors[a].UpdateDx(imt1.tensors[a].GetDx());
                originIMT.tensors[a].forEach([&](int i, int j, int k, int l, double *t) {
                    *t = imt1.tensors[a](i, j, k, l) + imt2.tensors[a](i, j, k, l);
                });
            }
        }
        HOTRG::contractionY(D_cut, T, T, U, "bottom");
        delete[] U;
    }

    /* normalization */
    T.normalization(normalize_factor);
    if (distance) {
        REP(i, DIMENSION) {
            originIMT.tensors[i].normalization(normalize_factor);
            orders[i] += originIMT.tensors[i].order - T.order;
        }
    }
    REP(i, DIMENSION) {
        IMT.tensors[i].normalization(normalize_factor);
        orders[i] += IMT.tensors[i].order - T.order;
    }

    double Tr = T.trace();

    REP(i, DIMENSION) {
        double impureTr = IMT.tensors[i].trace();
        const long long int order = orders[i];
        const long long int absOrder = std::abs(order);
        if (order > 0) {
            REP(k, absOrder) impureTr *= normalize_factor;
        } else {
            REP(k, absOrder) impureTr /= normalize_factor;
        }
        res[i] = impureTr / Tr;
    }
}

void
HOTRG::renormalization::mass_v1(Tensor &T, ImpureTensor &IMT, long long *orders, const int &N, const int &n, const int &merge_point, int &x_count, int &y_count, const int &normalize_factor, double *res) {
    const int D_cut = T.GetD_max();
    if ((n % 2 && x_count < merge_point - 1) || y_count == N / 2) { // compress along x-axis
        cout << " compress along x-axis " << std::flush;
        x_count++;
        const int Dy = T.GetDy();
        auto U = new double[Dy * Dy * Dy * Dy];
        HOTRG::SVD_Y(D_cut, T, U);
        if (x_count == merge_point) {
            for (auto &tensor : IMT.tensors) {
                HOTRG::contractionX(D_cut, tensor, tensor, U, "left");
            }
        } else {
            for (auto &tensor : IMT.tensors) {
                HOTRG::contractionX(D_cut, tensor, T, U, "left");
            }
        }
        HOTRG::contractionX(D_cut, T, T, U, "left");
        delete[] U;
    } else { // compress along y-axis
        cout << " compress along y-axis " << std::flush;
        y_count++;
        const int Dx = T.GetDx();
        auto U = new double[Dx * Dx * Dx * Dx];
        HOTRG::SVD_X(D_cut, T, U);
        auto imt1 = IMT;
        auto imt2 = IMT;
        for (int a = 0; a < 3; ++a) {
            HOTRG::contractionY(D_cut, imt1.tensors[a], T, U, "bottom");
            HOTRG::contractionY(D_cut, T, imt2.tensors[a], U, "top");
            IMT.tensors[a].UpdateDx(imt1.tensors[a].GetDx());
            IMT.tensors[a].forEach([&](int i, int j, int k, int l, double *t) {
                *t = imt1.tensors[a](i, j, k, l) + imt2.tensors[a](i, j, k, l);
            });
        }
        HOTRG::contractionY(D_cut, T, T, U, "bottom");
        delete[] U;
    }

    /* normalization */
    T.normalization(normalize_factor);
    for (auto &tensor : IMT.tensors) tensor.normalization(normalize_factor);
    REP(i, DIMENSION) {
        long long int order = IMT.tensors[i].order - T.order;
        if (x_count < merge_point) {
            order *= 2;
        }
        orders[i] += order;
    }

    double Tr = T.trace();

    REP(i, DIMENSION) {
        double impureTr = IMT.tensors[i].trace();
        const long long int order = orders[i];
        const long long int absOrder = std::abs(order);
        if (order > 0) {
            REP(k, absOrder) impureTr *= normalize_factor;
        } else {
            REP(k, absOrder) impureTr /= normalize_factor;
        }
        res[i] = impureTr / Tr;
    }
}