#include "../include/spherical_harmonics.hpp"
#include <gsl/gsl_specfunc.h>
#include <cmath>

#define REP(i, N) for (int i = 0; i < (N); ++i)
#define REP4(i, j, k, l, N) REP(i, N) REP(j, N) REP(k, N) REP(l, N)

void SphericalHarmonics::initTensor(const double &K, const int &l_max, Tensor &T) {
    auto A = new double[l_max];
    REP(i, l_max) {
        A[i] = gsl_sf_bessel_Inu(i + 0.5, K) * (i * 2 + 1);
    }

    REP4(i, j, k, l, l_max) {
        for (int im = 0; im <= 2 * i; ++im)
            for (int jm = 0; jm <= 2 * j; ++jm)
                for (int km = 0; km <= 2 * k; ++km)
                    for (int lm = 0; lm <= 2 * l; ++lm) {
                        double sum = 0;
                        for (int L = std::abs(i - j); L <= i + j; ++L)
                            for (int M = -L; M <= L; ++M) {
                                double c = 1;
                                const int two_i = 2 * i, two_j = 2 * j, two_k = 2 * k, two_l = 2 * l;
                                const int two_im = 2 * (im - i), two_jm = 2 * (jm - j), two_km = 2 * (km - k), two_lm = 2 * (lm - l);
                                const int two_L = 2 * L, two_M = 2 * M;
                                c *= gsl_sf_coupling_3j(two_i, two_j, two_L, two_im, two_jm, -two_M);
                                c *= gsl_sf_coupling_3j(two_i, two_j, two_L, 0, 0, 0);
                                c *= gsl_sf_coupling_3j(two_k, two_l, two_L, two_km, two_lm, -two_M);
                                c *= gsl_sf_coupling_3j(two_k, two_l, two_L, 0, 0, 0);
                                sum += (2 * L + 1) * c;
                            }
                        T(i * i + im, j * j + jm, k * k + km, l * l + lm) = std::sqrt(A[i] * A[j] * A[k] * A[l]) * sum;
                    }
    }
    delete[] A;
}

void SphericalHarmonics::initTensorWithImpure(const double &K, const int &l_max, Tensor &T, ImpureTensor &IMT) {
    initTensor(K, l_max, T);
    auto A = new double[l_max];
    REP(i, l_max) {
        A[i] = gsl_sf_bessel_Inu(i + 0.5, K) * (i * 2 + 1);
    }

    REP4(i, j, k, l, l_max) {
                    for (int im = 0; im <= 2 * i; ++im)
                        for (int jm = 0; jm <= 2 * j; ++jm)
                            for (int km = 0; km <= 2 * k; ++km)
                                for (int lm = 0; lm <= 2 * l; ++lm) {
                                    double sum[3];
                                    sum[0] = 0, sum[1] = 0, sum[2] = 0;
                                    for (int L = std::abs(i - j); L <= i + j; ++L)
                                        for (int L_ = std::abs(k - l); L_ <= k + l; ++L_)
                                            for (int M = -L; M <= L; ++M)
                                                for (int M_ = -L_; M_ <= L_; ++M_) {
                                                    for (int m = 0; m < 3; ++m) {
                                                        double c = 1;
                                                        const int two_i = 2 * i, two_j = 2 * j, two_k = 2 * k, two_l = 2 * l;
                                                        const int two_im = 2 * (im - i), two_jm = 2 * (jm - j), two_km = 2 * (km - k), two_lm = 2 * (lm - l);
                                                        const int two_L = 2 * L, two_M = 2 * M, two_L_ = 2 * L_, two_M_ = 2 * M_, two_m = 2 * (m - 1);
                                                        c *= gsl_sf_coupling_3j(two_i, two_j, two_L, two_im, two_jm, -two_M);
                                                        c *= gsl_sf_coupling_3j(two_i, two_j, two_L, 0, 0, 0);
                                                        c *= gsl_sf_coupling_3j(two_k, two_l, two_L_, two_km, two_lm, -two_M_);
                                                        c *= gsl_sf_coupling_3j(two_k, two_l, two_L_, 0, 0, 0);
                                                        c *= gsl_sf_coupling_3j(two_L_, 2, two_L, two_M_, two_m, -two_M);
                                                        c *= gsl_sf_coupling_3j(two_L_, 2, two_L, 0, 0, 0);
                                                        sum[m] += c * (2 * L + 1) * (2 * L_ + 1) * (1 - 2 * (std::abs(M_) % 2));
                                                    }
                                                }
                                    double a = std::sqrt(A[i] * A[j] * A[k] * A[l]);
                                    IMT.tensors[0](i * i + im, j * j + jm, k * k + km, l * l + lm) = a * (sum[0] - sum[2]) / std::sqrt(2);
                                    IMT.tensors[1](i * i + im, j * j + jm, k * k + km, l * l + lm) = a * (sum[0] + sum[2]) / std::sqrt(2);
                                    IMT.tensors[2](i * i + im, j * j + jm, k * k + km, l * l + lm) = a * sum[1];
                                }
                }
    delete[] A;
}