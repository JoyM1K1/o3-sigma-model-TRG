//
// Created by Joy on 2020/06/03.
//

#include <iostream>
#include "../include/legendre_zero_point.hpp"
#include <cmath>
#include <algorithm>
#include <gsl/gsl_specfunc.h>

#define ERROR 10e-6

double math::solver::d_legendre(int n, double x) {
    if (n == 0) return 0;
    return n * (gsl_sf_legendre_Pl(n - 1, x) - x * gsl_sf_legendre_Pl(n, x)) / (1 - x * x);
}

double math::solver::newton(int n, double x) {
    // 最大100回まで試す
    for (int i = 0; i < 100; ++i) {
        double d = d_legendre(n, x);
        if (d == 0) {
            std::cerr << "Error : derivative is zero\n";
            exit(1);
        }
        x -= gsl_sf_legendre_Pl(n, x) / d;
        if (std::abs(gsl_sf_legendre_Pl(n, x)) < ERROR) {
            return x;
        }
    }
    // 100回試してダメならエラー
    std::cerr << "Error : can't find zero point\n";
    exit(1);
}

std::vector<double> math::solver::legendre_zero_point(int n) {
    std::vector<double> ans(n);
    for (int i = 0; i < n; ++i) {
        // 初期値
        double init = std::cos((i + 0.75) / (n + 0.5) * M_PI);
        // newton法
        ans[i] = newton(n, init);
    }
    std::sort(ans.begin(), ans.end());
    return ans;
}