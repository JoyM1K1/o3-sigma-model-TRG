//
// Created by Joy on 2020/05/31.
//

#include <iomanip>
#include <iostream>
#include <cmath>
#include <legendre_zero_point.hpp>
#include <gsl/gsl_specfunc.h>

#define N 100

int main() {
    std::cout << std::setprecision(16);
    std::vector<double> v = math::solver::legendre_zero_point(N);
    for (int i = 0; i < N; ++i) {
        std::cout << v[i] << ' ';
    }
    std::cout << '\n';
    std::vector<double> p(N);
    for (int i = 0; i < N; ++i) {
        p[i] = gsl_sf_legendre_Pl(N - 1, v[i]);
    }
    std::vector<double> w(N);
    for (int i = 0; i < N; ++i) {
        w[i] = 2 * (1 - v[i] * v[i]) / (N * N * p[i] * p[i]);
        std::cout << w[i] << ' ';
    }
    std::cout << '\n';
}