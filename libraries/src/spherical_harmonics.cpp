#include "../include/spherical_harmonics.hpp"
#include <gsl/gsl_specfunc.h>
#include <cmath>
#include <iostream>
#include "../include/time_counter.hpp"

#define REP(i, N) for (int i = 0; i < (N); ++i)

void SphericalHarmonics::init_tensor(const double &beta, const int &l_max, BaseTensor &T) {
    auto A = new double[l_max + 1];
    REP(i, l_max + 1) {
        A[i] = gsl_sf_bessel_Inu(i + 0.5, beta) * (i * 2 + 1);
    }

    int n;
    int i, j, k, l;
    int two_i, two_j, two_k, two_l;
    int im, jm, km, lm;
    int two_im, two_jm, two_km, two_lm;
    int L;
    int two_L, two_M;
    double sum, a;
    double c;
    const int components_num = (l_max + 1) * (l_max + 1) * (l_max + 1) * (l_max + 1);

#pragma omp parallel for default(none) private(n, i, j, k, l, two_i, two_j, two_k, two_l, im, jm, km, lm, two_im, two_jm, two_km, two_lm, L, two_L, two_M, c, sum, a) shared(beta, l_max, A, T, components_num, std::cerr) schedule(static, 1)
    for (n = 0; n < components_num; ++n) {
        i = n % (l_max + 1);
        j = n / (l_max + 1) % (l_max + 1);
        k = n / (l_max + 1) / (l_max + 1) % (l_max + 1);
        l = n / (l_max + 1) / (l_max + 1) / (l_max + 1) % (l_max + 1);
        for (im = -i; im <= i; ++im)
            for (jm = -j; jm <= j; ++jm)
                for (km = -k; km <= k; ++km)
                    for (lm = -l; lm <= l; ++lm) {
                        sum = 0;
                        if (im + jm == km + lm) {
                            for (L = std::abs(i - j); L <= i + j; ++L) {
                                if (L < std::abs(k - l) || k + l < L) continue;
                                c = 1;
                                two_i = 2 * i, two_j = 2 * j, two_k = 2 * k, two_l = 2 * l;
                                two_im = 2 * im, two_jm = 2 * jm, two_km = 2 * km, two_lm = 2 * lm;
                                two_L = 2 * L, two_M = two_im + two_jm;
                                c *= gsl_sf_coupling_3j(two_i, two_j, two_L, two_im, two_jm, -two_M);
                                c *= gsl_sf_coupling_3j(two_i, two_j, two_L, 0, 0, 0);
                                c *= gsl_sf_coupling_3j(two_k, two_l, two_L, two_km, two_lm, -two_M);
                                c *= gsl_sf_coupling_3j(two_k, two_l, two_L, 0, 0, 0);
                                sum += (2 * L + 1) * c;
                            }
                        }
                        a = std::sqrt(A[i] * A[j] * A[k] * A[l]);
                        if (std::isnan(a)) {
                            std::cerr << '(' << i << ',' << im << ')'
                                      << '(' << j << ',' << jm << ')'
                                      << '(' << k << ',' << km << ')'
                                      << '(' << l << ',' << lm << ')'
                                      << " : a is nan\n";
                        }
                        if (std::isnan(sum)) {
                            std::cerr << '(' << i << ',' << im << ')'
                                      << '(' << j << ',' << jm << ')'
                                      << '(' << k << ',' << km << ')'
                                      << '(' << l << ',' << lm << ')'
                                      << " : sum is nan\n";
                        }
                        T(i * i + (i + im), j * j + (j + jm), k * k + (k + km), l * l + (l + lm)) = a * sum * M_PI / (2 * beta);
                    }
    }
    delete[] A;
}

template<class Tensor>
void SphericalHarmonics::init_tensor_with_impure(const double &beta, const int &l_max, Tensor &T, BaseImpureTensor<Tensor> &IMT) {
    auto A = new double[l_max + 1];
    REP(i, l_max + 1) {
        A[i] = gsl_sf_bessel_Inu(i + 0.5, beta) * (i * 2 + 1);
    }

    int n;
    int i, j, k, l;
    int index_i, index_j, index_k, index_l;
    int two_i, two_j, two_k, two_l;
    int im, jm, km, lm;
    int two_im, two_jm, two_km, two_lm;
    int L, L_, M, M_;
    int two_L, two_M, two_L_, two_M_;
    double sum[DIMENSION], s;
    double c, a;
    int m, two_m;
    const int components_num = (l_max + 1) * (l_max + 1) * (l_max + 1) * (l_max + 1);

#pragma omp parallel for default(none) private(n, i, j, k, l, two_i, two_j, two_k, two_l, im, jm, km, lm, two_im, two_jm, two_km, two_lm, L, L_, M, M_, two_L, two_M, two_L_, two_M_, m, two_m, c, sum, s, a, index_i, index_j, index_k, index_l) shared(beta, l_max, A, T, IMT, components_num, std::cerr) schedule(static, 1)
    for (n = 0; n < components_num; ++n) {
        i = n % (l_max + 1);
        j = n / (l_max + 1) % (l_max + 1);
        k = n / (l_max + 1) / (l_max + 1) % (l_max + 1);
        l = n / (l_max + 1) / (l_max + 1) / (l_max + 1) % (l_max + 1);
        two_i = 2 * i, two_j = 2 * j, two_k = 2 * k, two_l = 2 * l;
        for (im = -i; im <= i; ++im)
            for (jm = -j; jm <= j; ++jm)
                for (km = -k; km <= k; ++km)
                    for (lm = -l; lm <= l; ++lm) {
                        M = im + jm, M_ = km + lm;
                        two_im = 2 * im, two_jm = 2 * jm, two_km = 2 * km, two_lm = 2 * lm;
                        two_M = 2 * M, two_M_ = 2 * M_;
                        sum[0] = 0, sum[1] = 0, sum[2] = 0, s = 0;
                        /* pure tensor */
                        for (L = std::abs(i - j); L <= i + j; ++L) {
                            two_L = 2 * L;
                            c = 1;
                            c *= gsl_sf_coupling_3j(two_i, two_j, two_L, two_im, two_jm, -two_M);
                            c *= gsl_sf_coupling_3j(two_i, two_j, two_L, 0, 0, 0);
                            c *= gsl_sf_coupling_3j(two_k, two_l, two_L, two_km, two_lm, -two_M);
                            c *= gsl_sf_coupling_3j(two_k, two_l, two_L, 0, 0, 0);
                            s += (2 * L + 1) * c;
                        }
                        /* impure tensor */
                        for (m = -1; m <= 1; ++m) {
                            if (M == M_ + m) {
                                for (L = std::abs(i - j); L <= i + j; ++L)
                                    for (L_ = std::abs(k - l); L_ <= k + l; ++L_) {
                                        c = 1;
                                        two_L = 2 * L, two_L_ = 2 * L_, two_m = 2 * m;
                                        c *= gsl_sf_coupling_3j(two_i, two_j, two_L, two_im, two_jm, -two_M);
                                        c *= gsl_sf_coupling_3j(two_i, two_j, two_L, 0, 0, 0);
                                        c *= gsl_sf_coupling_3j(two_k, two_l, two_L_, two_km, two_lm, -two_M_);
                                        c *= gsl_sf_coupling_3j(two_k, two_l, two_L_, 0, 0, 0);
                                        c *= gsl_sf_coupling_3j(two_L_, 2, two_L, two_M_, two_m, -two_M);
                                        c *= gsl_sf_coupling_3j(two_L_, 2, two_L, 0, 0, 0);
                                        if ((int) std::abs(M_) % 2) c *= -1;
                                        sum[1 + m] += c * (2 * L + 1) * (2 * L_ + 1);
                                    }
                            }
                        }
                        a = std::sqrt(A[i] * A[j] * A[k] * A[l]);
                        if (std::isnan(a)) {
                            std::cerr << '(' << i << ',' << im << ')'
                                      << '(' << j << ',' << jm << ')'
                                      << '(' << k << ',' << km << ')'
                                      << '(' << l << ',' << lm << ')'
                                      << " : a is nan\n";
                        }
                        for (m = 0; m < 3; ++m) {
                            if (std::isnan(sum[m])) {
                                std::cerr << '(' << i << ',' << im << ')'
                                          << '(' << j << ',' << jm << ')'
                                          << '(' << k << ',' << km << ')'
                                          << '(' << l << ',' << lm << ')'
                                          << " : sum[" << m << "] is nan\n";
                            }
                        }
                        if (std::isnan(s)) {
                            std::cerr << '(' << i << ',' << im << ')'
                                      << '(' << j << ',' << jm << ')'
                                      << '(' << k << ',' << km << ')'
                                      << '(' << l << ',' << lm << ')'
                                      << " : s is nan\n";
                        }
                        index_i = i * i + (i + im), index_j = j * j + (j + jm), index_k = k * k + (k + km), index_l = l * l + (l + lm);
                        T(index_i, index_j, index_k, index_l) = a * s * M_PI / (2 * beta);
                        IMT.tensors[0](index_i, index_j, index_k, index_l) = a * (sum[0] - sum[2]) / std::sqrt(2) * M_PI / (2 * beta);
                        IMT.tensors[1](index_i, index_j, index_k, index_l) = a * (sum[0] + sum[2]) / std::sqrt(2) * M_PI / (2 * beta);
                        IMT.tensors[2](index_i, index_j, index_k, index_l) = a * sum[1] * M_PI / (2 * beta);
                    }
    }
    delete[] A;
}

template void SphericalHarmonics::init_tensor_with_impure(const double &beta, const int &l_max, TRG::Tensor &T, BaseImpureTensor<TRG::Tensor> &IMT);

template void SphericalHarmonics::init_tensor_with_impure(const double &beta, const int &l_max, HOTRG::Tensor &T, BaseImpureTensor<HOTRG::Tensor> &IMT);