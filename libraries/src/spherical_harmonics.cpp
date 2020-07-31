#include "../include/spherical_harmonics.hpp"
#include <gsl/gsl_specfunc.h>
#include <cmath>
#include <iostream>

#define REP(i, N) for (int i = 0; i < (N); ++i)
#define REP4(i, j, k, l, N) REP(i, N) REP(j, N) REP(k, N) REP(l, N)

void SphericalHarmonics::initTensor(const double &K, const int &l_max, Tensor &T) {
    auto A = new double[l_max + 1];
    REP(i, l_max + 1) {
        A[i] = gsl_sf_bessel_Inu(i + 0.5, K) * (i * 2 + 1);
    }

    int n;
    int i, j, k, l;
    int two_i, two_j, two_k, two_l;
    int im, jm, km, lm;
    int two_im, two_jm, two_km, two_lm;
    int L, M;
    int two_L, two_M;
    double sum, a;
    double c;

#pragma omp parallel for default(none) private(n, i, j, k, l, two_i, two_j, two_k, two_l, im, jm, km, lm, two_im, two_jm, two_km, two_lm, L, M, two_L, two_M, c, sum, a) shared(l_max, A, T, std::cerr) schedule(static, 1)
    for (n = 0; n < (l_max + 1) * (l_max + 1) * (l_max + 1) * (l_max + 1); ++n) {
        i = n % (l_max + 1);
        j = n / (l_max + 1) % (l_max + 1);
        k = n / (l_max + 1) / (l_max + 1) % (l_max + 1);
        l = n / (l_max + 1) / (l_max + 1) / (l_max + 1) % (l_max + 1);
        for (im = 0; im <= 2 * i; ++im)
            for (jm = 0; jm <= 2 * j; ++jm)
                for (km = 0; km <= 2 * k; ++km)
                    for (lm = 0; lm <= 2 * l; ++lm) {
                        sum = 0;
                        for (L = std::abs(i - j); L <= i + j; ++L)
                            for (M = -L; M <= L; ++M) {
                                if (L < std::abs(k - l) || k + l < L || im - i + jm - j != M || km - k + lm - l != M) continue;
                                c = 1;
                                two_i = 2 * i, two_j = 2 * j, two_k = 2 * k, two_l = 2 * l;
                                two_im = 2 * (im - i), two_jm = 2 * (jm - j), two_km = 2 * (km - k), two_lm = 2 * (lm - l);
                                two_L = 2 * L, two_M = 2 * M;
                                c *= gsl_sf_coupling_3j(two_i, two_j, two_L, two_im, two_jm, -two_M);
                                c *= gsl_sf_coupling_3j(two_i, two_j, two_L, 0, 0, 0);
                                c *= gsl_sf_coupling_3j(two_k, two_l, two_L, two_km, two_lm, -two_M);
                                c *= gsl_sf_coupling_3j(two_k, two_l, two_L, 0, 0, 0);
                                sum += (2 * L + 1) * c;
                            }
                        a = std::sqrt(A[i] * A[j] * A[k] * A[l]);
                        if (std::isnan(a)) {
                            std::cerr << '(' << i << ',' << i - im << ')'
                                      << '(' << j << ',' << j - jm << ')'
                                      << '(' << k << ',' << k - km << ')'
                                      << '(' << l << ',' << l - lm << ')'
                                      << " : a is nan\n";
                        }
                        if (std::isnan(sum)) {
                            std::cerr << '(' << i << ',' << i - im << ')'
                                      << '(' << j << ',' << j - jm << ')'
                                      << '(' << k << ',' << k - km << ')'
                                      << '(' << l << ',' << l - lm << ')'
                                      << " : sum is nan\n";
                        }
//                        if (std::isnan(a * sum)) {
//                            std::cerr << '(' << i << ',' << i - im << ')'
//                                      << '(' << j << ',' << j - jm << ')'
//                                      << '(' << k << ',' << k - km << ')'
//                                      << '(' << l << ',' << l - lm << ')'
//                                      << " : a * sum is nan\n";
//                        }
                        T(i * i + im, j * j + jm, k * k + km, l * l + lm) = a * sum;
                    }
    }
    delete[] A;
}

void SphericalHarmonics::initTensorWithImpure(const double &K, const int &l_max, Tensor &T, ImpureTensor &IMT) {
    auto A = new double[l_max + 1];
    REP(i, l_max + 1) {
        A[i] = gsl_sf_bessel_Inu(i + 0.5, K) * (i * 2 + 1);
    }

    int n;
    int i, j, k, l;
    int two_i, two_j, two_k, two_l;
    int im, jm, km, lm;
    int two_im, two_jm, two_km, two_lm;
    int L, M, L_, M_;
    int two_L, two_M, two_L_, two_M_;
    double sum[3], s;
    double c, a;
    int m, two_m;

#pragma omp parallel for default(none) private(n, i, j, k, l, two_i, two_j, two_k, two_l, im, jm, km, lm, two_im, two_jm, two_km, two_lm, L, M, L_, M_, two_L, two_M, two_L_, two_M_, m, two_m, c, sum, s, a) shared(l_max, A, T, IMT, std::cerr) schedule(static, 1)
    for (n = 0; n < (l_max + 1) * (l_max + 1) * (l_max + 1) * (l_max + 1); ++n) {
        i = n % (l_max + 1);
        j = n / (l_max + 1) % (l_max + 1);
        k = n / (l_max + 1) / (l_max + 1) % (l_max + 1);
        l = n / (l_max + 1) / (l_max + 1) / (l_max + 1) % (l_max + 1);
        for (im = 0; im <= 2 * i; ++im)
            for (jm = 0; jm <= 2 * j; ++jm)
                for (km = 0; km <= 2 * k; ++km)
                    for (lm = 0; lm <= 2 * l; ++lm) {
                        sum[0] = 0, sum[1] = 0, sum[2] = 0, s = 0;
                        for (L = std::abs(i - j); L <= i + j; ++L)
                            for (L_ = std::abs(k - l); L_ <= k + l; ++L_)
                                for (M = -L; M <= L; ++M)
                                    for (M_ = -L_; M_ <= L_; ++M_)
                                        for (m = 0; m < 3; ++m) {
                                            c = 1;
                                            two_i = 2 * i, two_j = 2 * j, two_k = 2 * k, two_l = 2 * l;
                                            two_im = 2 * (im - i), two_jm = 2 * (jm - j), two_km = 2 * (km - k), two_lm = 2 * (lm - l);
                                            two_L = 2 * L, two_M = 2 * M, two_L_ = 2 * L_, two_M_ = 2 * M_, two_m = 2 * (m - 1);
                                            c *= gsl_sf_coupling_3j(two_i, two_j, two_L, two_im, two_jm, -two_M);
                                            c *= gsl_sf_coupling_3j(two_i, two_j, two_L, 0, 0, 0);
                                            c *= gsl_sf_coupling_3j(two_k, two_l, two_L_, two_km, two_lm, -two_M_);
                                            c *= gsl_sf_coupling_3j(two_k, two_l, two_L_, 0, 0, 0);
                                            if (L == L_ && M == M_) {
                                                s += (2 * L + 1) * c;
                                            }
                                            c *= gsl_sf_coupling_3j(two_L_, 2, two_L, two_M_, two_m, -two_M);
                                            c *= gsl_sf_coupling_3j(two_L_, 2, two_L, 0, 0, 0);
                                            if ((int) std::abs(M_) % 2) c *= -1;
                                            sum[m] += c * (2 * L + 1) * (2 * L_ + 1);
                                        }
                        a = std::sqrt(A[i] * A[j] * A[k] * A[l]);
                        if (std::isnan(a)) {
                            std::cerr << '(' << i << ',' << i - im << ')'
                                      << '(' << j << ',' << j - jm << ')'
                                      << '(' << k << ',' << k - km << ')'
                                      << '(' << l << ',' << l - lm << ')'
                                      << " : a is nan\n";
                        }
                        for (m = 0; m < 3; ++m) {
                            if (std::isnan(sum[m])) {
                                std::cerr << '(' << i << ',' << i - im << ')'
                                          << '(' << j << ',' << j - jm << ')'
                                          << '(' << k << ',' << k - km << ')'
                                          << '(' << l << ',' << l - lm << ')'
                                          << " : sum[" << m << "] is nan\n";
                            }
                        }
                        if (std::isnan(s)) {
                            std::cerr << '(' << i << ',' << i - im << ')'
                                      << '(' << j << ',' << j - jm << ')'
                                      << '(' << k << ',' << k - km << ')'
                                      << '(' << l << ',' << l - lm << ')'
                                      << " : s is nan\n";
                        }
                        T(i * i + im, j * j + jm, k * k + km, l * l + lm) = a * s;
                        IMT.tensors[0](i * i + im, j * j + jm, k * k + km, l * l + lm) = a * (sum[0] - sum[2]) / std::sqrt(2);
                        IMT.tensors[1](i * i + im, j * j + jm, k * k + km, l * l + lm) = a * (sum[0] + sum[2]) / std::sqrt(2);
                        IMT.tensors[2](i * i + im, j * j + jm, k * k + km, l * l + lm) = a * sum[1];
                    }
    }

    delete[] A;
}