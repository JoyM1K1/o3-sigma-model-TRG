//
// Created by Joy on 2020/05/15.
//

#ifndef O3_SIGMA_MODEL_LEGENDRE_ZERO_POINT_HPP
#define O3_SIGMA_MODEL_LEGENDRE_ZERO_POINT_HPP

#include <cstddef>
#include <vector>
#include <cmath>
#include <functional>
#include <iostream>

#define ERROR 10e-6

namespace math::solver {
    // P_n(x) の微分
    double d_legendre(size_t n, double x) {
        if (n == 0) return 0;
        return n * (std::legendre(n - 1, x) - x * std::legendre(n, x)) / (1 - x * x);
    }

    // newton法
    double newton(size_t n, double x) {
        // 最大100回まで試す
        for (size_t i = 0; i < 100; ++i) {
            double d = d_legendre(n, x);
            if (d == 0) {
                std::cerr << "Error : derivative is zero\n";
                exit(1);
            }
            x -= std::legendre(n, x) / d;
            if (std::abs(std::legendre(n, x)) < ERROR) {
                return x;
            }
        }
        // 100回試してダメならエラー
        std::cerr << "Error : can't find zero point\n";
        exit(1);
    }

    // newton法を用いてn次Legendre多項式のゼロ点を計算する
    std::vector<double> legendre_zero_point(size_t n) {
        std::vector<double> ans(n);
        for (size_t i = 0; i < n; ++i) {
            // 初期値
            double init = std::cos((i + 0.75) / (n + 0.5) * M_PI);
            // newton法
            ans[i] = newton(n, init);
        }
        std::sort(ans.begin(), ans.end());
        return ans;
    }
}

#endif //O3_SIGMA_MODEL_LEGENDRE_ZERO_POINT_HPP
